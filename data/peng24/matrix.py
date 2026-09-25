# %%
import polars as pl
from pathlib import Path

project_root = Path("../..").resolve()
print(f"Polars Thread Pool Size: {pl.thread_pool_size()}")

# %% [markdown]
# ## Functions

# %%
def select_standard_contigs(
	df: pl.DataFrame,
	chrom_col: str = "chrom",
	include_mito: bool = True,
) -> pl.DataFrame:
	"""
	Filter a DataFrame down to only the standard chromosomes (1-22, X, Y, [M]).
	
	Handles both 'chr'-prefixed (e.g. 'chr1') and bare (e.g. '1') naming case-insensitively.

	Parameters
	----------
	df : pl.DataFrame
	chrom_col : str
		Name of the chromosome column to filter on.
	include_mito : bool
		Whether to keep the mitochondrial contig (matches 'M' or 'MT').

	Returns
	-------
	pl.DataFrame
		Rows where `chrom_col` is a standard autosome/X/Y/[M].
	"""
	standard = [str(i) for i in range(1, 23)] + ["X", "Y"]
	if include_mito:
		standard += ["M", "MT"]

	normalized = pl.col(chrom_col).str.replace(r"(?i)^chr", "").str.to_uppercase()

	return df.filter(normalized.is_in(standard))

# %%
def human_sort_contigs(
	df: pl.DataFrame,
	by: list[str] | str = ["chrom", "pos"],
	chrom_col: str = "chrom",
	standard_chroms: list[str] | None = None,
	descending: bool | list[bool] = False,
) -> pl.DataFrame:
	"""
	Sort by `by`, using natural chromosome order. If chrom column is not present
	in `by`,  the function acts like the uusal polars sort, i.e. `df.sort(by=...)`. 

	Standard chromosomes gets sorted in order -> (1..22, X, Y, M, MT).
	scaffolds/decoys/unplaced contigs sort alphabetically after them. 
	"""
	## Check/format arguments
	by_list = [by] if isinstance(by, str) else list(by)

	desc_list = descending if isinstance(descending, list) else [descending] * len(by_list)
	if len(desc_list) != len(by_list):
		raise ValueError("`descending` list must be the same length as `by`.")

	# List of contigs which are considered the standard chromosomes
	# If not provided, the hg38 format is used i.e chr1, chr2 ... chrX, chrY, chrM
	if chrom_col in by_list:
		if standard_chroms is None:
			standard_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM", "chrMT"]

		# Contigs not present in standard chroms are put last
		present = df.get_column(chrom_col).unique().to_list()
		scaffolds = sorted(c for c in present if c not in standard_chroms)

		categories = standard_chroms + scaffolds
		# Chrom column is casted to categorical encoding facored by order of appearence in `categories`
		df = df.with_columns(pl.col(chrom_col).cast(pl.Enum(categories)))

	return df.sort(by=by_list, descending=desc_list)

# %%
def read_svcf(svcf_path: Path, standard_contigs: bool = False, sort: bool = True, columns: list = ["#CHROM", "POS", "REF", "ALT", "INFO"]) -> pl.DataFrame:
	svcf = (
		pl.read_csv(
			svcf_path,
			separator="\t",
			comment_prefix="##",
			null_values=".",
			columns=columns,
			truncate_ragged_lines=True,
		)
		.rename(lambda x: x.lstrip("#").lower())
	)

	if standard_contigs:
		svcf = select_standard_contigs(svcf, chrom_col="chrom")


	# --- pull end, svtype, svlen, sources, and source_ids out of the info string ---
	svcf = svcf.with_columns(
		pl.col("info").str.extract(r"END=([^;]+)").cast(pl.Int64).alias("end"),
		pl.col("info").str.extract(r"SVLEN=([^;]+)").replace(".", None).cast(pl.Int64).alias("svlen"),
		pl.col("info").str.extract(r"SVTYPE=([^;]+)").alias("svtype"),
		pl.col("info").str.extract(r"SOURCES=([^;]+)").alias("source"),
		# pl.col("info").str.extract(r"SOURCE_IDS=([^;]+)").alias("source_id"),
	)

	if sort:
		svcf = human_sort_contigs(svcf, chrom_col="chrom")

	return svcf

# %%
def annotate_svcf(svcf: pl.DataFrame, presence_value: bool = True) -> pl.DataFrame:
	## Peng24 VCF label pattern: {platform}_{sample}_{aligner}_{caller}_{sample}
	label_pattern = r"^(?<platform>[^_]+)_(?<sample>[^_]+)_(?<aligner>[^_]+)_(?<caller>[^_]+)_(?<sample2>[^_]+)$"

	return (
		svcf
		.with_columns(pl.col("source").str.split(","))
		.explode("source")
		.with_columns(pl.col("source").str.extract_groups(label_pattern).alias("parts"))
		.drop("source")
		.unnest("parts")
		.with_columns(pl.col("caller").str.to_lowercase())
		.with_columns(pl.lit(presence_value).alias("present"))
	)

# %%
def svcf2matrix(
	svcf_pass_path: Path, 
	svcf_all_path: Path,
	standard_contigs: bool = True,
	sort: bool = True,
) -> pl.DataFrame:

	if svcf_pass_path.stem != svcf_all_path.stem:
		raise ValueError (
			"Provided svcf_pass_path and svcf_all_path stems don't match. "
			f"svcf_pass_path.stem = {svcf_pass_path.stem}\n"
			f"svcf_all_path.stem = {svcf_all_path.stem}\n"
		)

	svcf_pass = read_svcf(svcf_pass_path, standard_contigs=standard_contigs, sort=sort)
	svcf_all = read_svcf(svcf_all_path, standard_contigs=standard_contigs, sort=sort)

	# SVs which only appear in the merge without pass filtering
	non_pass_svs = svcf_all.join(svcf_pass, on = ["chrom", "pos", "end", "ref", "alt", "svtype"], how="anti", nulls_equal=True)
	del svcf_all

	svcf_annotated = pl.concat([
		svcf_pass.pipe(annotate_svcf, presence_value=True), 
		non_pass_svs.pipe(annotate_svcf, presence_value=False)
	])
	del svcf_pass, non_pass_svs

	# Pivot to wide boolean matrix
	# The IDs defines locus identity for the matrix; columns omitted here gets silently dropped by the pivot
	id_cols = ["chrom", "pos", "end", "ref", "alt", "svtype", "svlen", "platform", "sample", "aligner"]

	## Check if there are multiple values contributing towards presence in a single caller' locus.
	collisions = (
		svcf_annotated.group_by(id_cols + ["caller"])
		.len()
		.filter(pl.col("len") > 1)
	)
	if collisions.height:
		print(
			f"WARNING: {collisions.height} (ids + caller) groups in {svcf_pass_path.stem} had multiple contributing source rows. \n" 
			"			Pivot's aggregate_function will collapse these into 1 boolean label (via OR) as a safety net."
		)

	# Caller columns are boolean indicating a call (TRUE) only if their FILTER was PASS, else FALSE
	matrix = svcf_annotated.pivot(
		values="present",
		index=id_cols,
		on="caller",
		aggregate_function=pl.element().any(),
	)

	# Fill Null caller columns with False
	caller_cols = [c for c in matrix.columns if c not in id_cols]
	matrix = matrix.with_columns([pl.col(c).fill_null(False) for c in caller_cols])

	if sort:
		matrix = human_sort_contigs(matrix, by=["chrom", "pos"], chrom_col="chrom")

	return matrix

# %% [markdown]
# ## Setup

# %%
annot_root = project_root / "annot"
data_root = project_root / "data"

dataset = "peng24"

matrix_outdir = data_root / dataset / "matrix"
matrix_outdir.mkdir(exist_ok=True, parents=True)

# octopusv union merge svcfs with only PASS SV
svcf_pass_paths = sorted(list((data_root / dataset / "merged_pass").glob("*.merged.svcf")))

# octopusv union merge svcfs with all SVs (retains non-PASS SVs)
svcf_all_paths = sorted(list((data_root / dataset / "merged_all").glob("*.merged.svcf")))

# %% [markdown]
# ## Main

# %%
for i, (svcf_pass_path, svcf_all_path) in enumerate(zip(svcf_pass_paths, svcf_all_paths)):
	cluster = svcf_pass_path.name.split(".")[0]
	svcf_all_path = svcf_all_paths[i]

	print(f"{i+1}. Processing octopuSV union merge cluster: {cluster}" )

	matrix = svcf2matrix(svcf_pass_path=svcf_pass_path, svcf_all_path=svcf_all_path, standard_contigs=True, sort=True)
	
	matrix.write_parquet(matrix_outdir / f"{cluster}.matrix.parquet",  compression="snappy")
