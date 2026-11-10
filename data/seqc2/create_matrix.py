# %%
import logging
import polars as pl
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("svcf2matrix")

project_root = Path("../..").resolve()

## Manifest column used to name matrix columns
LEVEL_COLS = {"platform": "platform", "caller": "caller", "file": "name"}

## Locus description carried into the matrix. Row identity is sv_idx (one row per merged record),
## so records that happen to share these values are never collapsed together.
ID_COLS = ["chrom", "pos", "end", "chr2", "ref", "alt", "svtype", "svlen", "id"]

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
def read_svcf(
	svcf_path: Path,
	standard_contigs: bool = False,
	sort: bool = True,
	columns: list = ["#CHROM", "POS", "ID", "REF", "ALT", "INFO"],
) -> pl.DataFrame:
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

	## (^|;) anchors stop END= from matching CIEND= / SVEND= etc.
	def info(key: str) -> pl.Expr:
		return pl.col("info").str.extract(rf"(?:^|;){key}=([^;]+)")

	svcf = svcf.with_columns(
		pl.col("ref").fill_null("N"),
		info("END").cast(pl.Int64, strict=False).alias("end"),
		info("SVLEN").cast(pl.Int64, strict=False).alias("svlen"),   # "." -> null
		info("SVTYPE").alias("svtype"),
		info("CHR2").alias("chr2"),
		info("SOURCES").alias("source"),
	)

	if sort:
		svcf = human_sort_contigs(svcf, chrom_col="chrom")

	return svcf

# %%
def read_manifest(path: Path) -> pl.DataFrame:
	m = pl.read_csv(path, separator="\t", schema_overrides={"split": pl.Utf8, "replicate": pl.Utf8})
	missing = {"label", "kind", "platform", "name", "split"} - set(m.columns)
	if missing:
		raise ValueError(f"{path.name}: manifest missing columns {missing}")
	if m["label"].is_duplicated().any():
		raise ValueError(f"{path.name}: duplicate labels")
	return m

# %%
def svcf2matrix(
	svcf_path: Path,
	manifest_path: Path,
	level: str = "platform",          # "platform" | "file"
	standard_contigs: bool = True,
	sort: bool = True,
	keep_truth_only: bool = True,
) -> pl.DataFrame:
	"""
	One row per merged record. Each platform (or file) column is True only if that
	platform contributed a PASS call to the record; a non-PASS call and no call are both
	False. Truth-set columns are True when the truth set supports the record.

	keep_truth_only: keep records supported only by truth sets (no platform call at all).
	These are platform false negatives and are needed to compute recall against the truth.
	"""
	if level not in LEVEL_COLS:
		raise ValueError(f"level must be one of {list(LEVEL_COLS)}")
	group_col = LEVEL_COLS[level]

	manifest = read_manifest(manifest_path).with_columns(
		pl.col(group_col).str.to_lowercase().alias("column")
	)
	if group_col not in manifest.columns:
		raise ValueError(f"manifest has no '{group_col}' column (re-run the merge script)")
	n_groups = manifest.select(group_col).n_unique()
	if manifest.select("column").n_unique() != n_groups:
		raise ValueError(f"'{group_col}' values collide after lowercasing")
	ambiguous = (
		manifest.filter(pl.col("kind") == "call")
		.group_by("column").agg(pl.col("platform").n_unique().alias("n"))
		.filter(pl.col("n") > 1)
	)
	if ambiguous.height:
		raise ValueError(
			f"level='{level}' columns span several platforms: {ambiguous['column'].to_list()}. "
			"Use level='caller' only on intra-platform merges, or level='file'."
		)

	svcf = (
		read_svcf(svcf_path, standard_contigs=standard_contigs, sort=False)
		.with_row_index("sv_idx")
	)
	if (n := svcf["source"].null_count()):
		raise ValueError(f"{n} records in {svcf_path.name} have no INFO/SOURCES")

	## Long format: one row per (record, contributing label)
	long = (
		svcf.select("sv_idx", pl.col("source").str.split(",").alias("label"))
		.explode("label")
		.join(manifest, on="label", how="left")
	)
	unknown = long.filter(pl.col("kind").is_null())["label"].unique().to_list()
	if unknown:
		raise ValueError(
			f"SOURCES labels not in manifest (stale manifest, or OctopuSV rewrote labels): {unknown[:10]}"
		)

	long = long.with_columns(
		((pl.col("kind") == "truth") | pl.col("split").eq("pass").fill_null(False)).alias("present")
	)

	## Diagnostics
	per_cell = (
		long.filter(pl.col("kind") == "call")
		.group_by("sv_idx", "column")
		.agg(pl.col("present").any().alias("any_pass"), pl.col("present").all().alias("all_pass"))
	)
	n_mixed = per_cell.filter(pl.col("any_pass") & ~pl.col("all_pass")).height
	if n_mixed:
		logger.info(
			f"{n_mixed} (record, {level}) cells have both PASS and non-PASS support -> True (any PASS wins)"
		)

	called = long.group_by("sv_idx").agg(
		(pl.col("kind") == "call").any().alias("called_by_platform")
	)

	## Wide boolean matrix; any() ORs multiple labels landing on the same column
	wide = long.pivot(
		on="column", index="sv_idx", values="present",
		aggregate_function=pl.element().any(),
	)

	## Platforms first, then truth sets, in manifest order; add all-False columns for absent groups
	col_order = manifest["column"].unique(maintain_order=True).to_list()
	wide = wide.with_columns([pl.lit(False).alias(c) for c in col_order if c not in wide.columns])

	matrix = (
		svcf.select("sv_idx", *ID_COLS)
		.join(called, on="sv_idx", how="left")
		.join(wide, on="sv_idx", how="left")
		.with_columns(pl.col(col_order).fill_null(False))
		.select("sv_idx", *ID_COLS, "called_by_platform", *col_order)
	)

	## Collapse rows at an identical locus. OctopuSV leaves these split when ALT differs
	## (TRA: truth '++' placeholder vs caller bracket orientation) or SVLEN is missing
	## (Bionano INV replicates). svlen stays in the key so different-size INS/DEL never merge.
	COLLAPSE_KEY = ["chrom", "pos", "end", "chr2", "svtype", "svlen"]
	n_before = matrix.height
	matrix = (
		matrix.group_by(COLLAPSE_KEY, maintain_order=True)
		.agg(
			pl.col("sv_idx").min(),
			pl.col("ref", "alt", "id").first(),
			pl.len().alias("n_collapsed"),
			pl.col("called_by_platform", *col_order).any(),
		)
		.select("sv_idx", *ID_COLS, "n_collapsed", "called_by_platform", *col_order)
	)
	logger.info(f"Collapsed {n_before - matrix.height} same-locus rows (TRA orientation / INV SVLEN splits)")

	logger.info(
		f"{matrix.height} records; True counts:\n"
		+ str(matrix.select(pl.col(col_order).sum()))
	)

	if sort:
		matrix = human_sort_contigs(matrix, by=["chrom", "pos"], chrom_col="chrom")

	return matrix

# %% [markdown]
# ## Setup

# %%
dataset = "seqc2"
data_root = project_root / "data" / dataset

matrix_outdir = data_root / "matrix"
matrix_outdir.mkdir(exist_ok=True, parents=True)

## Single flat OctopuSV union merge: every caller file x {pass, nonpass} + truth sets
svcf_path = data_root / "platforms.merged.split.svcf"
manifest_path = data_root / "platforms.merged.split.sources.tsv"

# %% [markdown]
# ## Main

# %%
matrix = svcf2matrix(svcf_path, manifest_path, level="platform", standard_contigs=True, sort=True)
matrix.write_parquet(matrix_outdir / "platforms.matrix.parquet", compression="snappy")

matrix_consensus = matrix.filter(pl.col("seqc2_consensus_svs"))
matrix_consensus.write_parquet(
	matrix_outdir / "platforms.seqc2_consensus_validated.matrix.parquet", compression="snappy"
)

## Optional: same rule, one column per caller file, from the same merge
# matrix_file = svcf2matrix(svcf_path, manifest_path, level="file")
# matrix_file.write_parquet(matrix_outdir / "files.matrix.parquet", compression="snappy")


# %% [markdown]
# ## Intra-platform (caller-level) matrices

# # %%
# platform_index = pl.read_csv(data_root / "intra-platform_merge" / "index.tsv", separator="\t")
# intra_outdir = matrix_outdir / "intra-platform"
# intra_outdir.mkdir(exist_ok=True, parents=True)

# for rec in platform_index.iter_rows(named=True):
# 	logger.info(f"== {rec['platform']} ({rec['callers']})")
# 	for level in ("caller", "file"):
# 		m = svcf2matrix(Path(rec["svcf"]), Path(rec["manifest"]), level=level,
# 		                standard_contigs=True, sort=True)
# 		m.write_parquet(intra_outdir / f"{rec['slug']}.{level}.matrix.parquet", compression="snappy")
# 		m.filter(pl.col("seqc2_consensus_svs")).write_parquet(
# 			intra_outdir / f"{rec['slug']}.{level}.seqc_consensus_validated.matrix.parquet",
# 			compression="snappy",
# 		)