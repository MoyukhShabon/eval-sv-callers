# %%
import polars as pl
from pathlib import Path
import os, tempfile, subprocess, logging, gzip, re

PROJECT_ROOT = Path("../..").resolve()

# %%
logging.basicConfig(
	level=logging.INFO,
	format="\n==[%(levelname)s | %(asctime)s]: %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	force=True
)
logger = logging.getLogger()

# %%
def run(cmd: list[str]) -> None:
	logger.info("=> " + " ".join(cmd))
	subprocess.run(cmd, check=True)

# %%
dataset = "seqc2"
DATA_ROOT = PROJECT_ROOT / f"data/{dataset}"
vcf_dir = DATA_ROOT / "vcf"

# %%
## SEQC2 alredy removed germline SVs from tumor VCF
annot = (
    pl.read_csv(PROJECT_ROOT / f"annot/{dataset}/vcf_annotations.tsv", separator="\t")
    .drop("site", "tissue_code", "replicate")
    .filter(pl.col("tissue") == "tumor")
)

# %%
## Standard SV types allowed by the VCF spec / supported by OctopuSV
VALID_SVTYPES = {"DEL", "DUP", "INS", "INV", "BND", "TRA"}
SVTYPE_RE = re.compile(r"(?:^|;)SVTYPE=([^;]+)")

# %%
def sanitize_vcf(in_vcf_gz: Path) -> tuple[Path, dict[str, int], list[dict]]:
	"""
	Decompress a VCF.gz file and drop any record whose SVTYPE is missing or
	not one of VALID_SVTYPES. Header lines (#) are passed through untouched.

	Returns:
		out_path: path to the sanitized, plain-text VCF
		svtype_counts: dict mapping each valid SVTYPE (+ "illegal") -> count of records seen
		dropped_records: list of dicts describing each dropped record
	"""
	fd, name = tempfile.mkstemp(suffix=".vcf")
	os.close(fd)
	out_path = Path(name)

	svtype_counts: dict[str, int] = {t: 0 for t in VALID_SVTYPES}
	svtype_counts["illegal"] = 0
	dropped_records: list[dict] = []

	try:
		with gzip.open(in_vcf_gz, "rt") as fin, open(out_path, "w") as fout:
			for line in fin:
				if line.startswith("#"):
					fout.write(line)
					continue

				fields = line.rstrip("\n").split("\t")
				chrom, pos, vid, info = fields[0], fields[1], fields[2], fields[7]

				m = SVTYPE_RE.search(info)
				svtype = m.group(1) if m else None

				if svtype is None or svtype not in VALID_SVTYPES:
					svtype_counts["illegal"] += 1
					dropped_records.append({
						"filename": in_vcf_gz.name,
						"chrom": chrom,
						"pos": pos,
						"id": vid,
						"illegal_svtype": svtype if svtype is not None else "MISSING",
					})
					continue

				svtype_counts[svtype] += 1
				fout.write(line)
	except BaseException:
		out_path.unlink(missing_ok=True)
		raise

	return out_path, svtype_counts, dropped_records

# %%
outdir_correct_all = DATA_ROOT / "octopusv_correct_all"
outdir_correct_pass = DATA_ROOT / "octopusv_correct_pass"
outdir_correct_all.mkdir(parents=True, exist_ok=True)
outdir_correct_pass.mkdir(parents=True, exist_ok=True)

vcf_dir = DATA_ROOT / "vcf"

file_totals_rows: list[dict] = []
illegal_sv_types_rows: list[dict] = []

for filename in annot["filename"]:
	in_vcf_gz = vcf_dir / filename
	name = filename.removesuffix(".vcf.gz")

	logger.info(f"Sanitizing & correcting: {name}")

	sanitized_path, svtype_counts, dropped_records = sanitize_vcf(in_vcf_gz)

	total_svs = sum(svtype_counts.values())
	file_totals_rows.append({
		"filename": name,
		**{t: svtype_counts[t] for t in VALID_SVTYPES},
		"illegal": svtype_counts["illegal"],
		"total_svs": total_svs,
	})

	illegal_counts: dict[str, int] = {}
	for rec in dropped_records:
		illegal_counts[rec["illegal_svtype"]] = illegal_counts.get(rec["illegal_svtype"], 0) + 1

	for illegal_svtype, n_dropped in illegal_counts.items():
		logger.info(f"WARNING [{name}]: dropped {n_dropped} record(s) with SVTYPE={illegal_svtype}")
		illegal_sv_types_rows.append({
			"filename": name,
			"illegal_svtype": illegal_svtype,
			"n_dropped": n_dropped,
		})

	try:
		outpath_all = outdir_correct_all / f"{name}.svcf"
		run(["octopusv", "correct", "-i", str(sanitized_path), "-o", str(outpath_all)])

		outpath_pass = outdir_correct_pass / f"{name}.svcf"
		run(["octopusv", "correct", "-i", str(sanitized_path), "-o", str(outpath_pass), "--filter-pass"])
	finally:
		sanitized_path.unlink(missing_ok=True)

# %%
## count of each valid SVTYPE + illegal count + total per VCF
file_totals = pl.DataFrame(
	file_totals_rows,
	schema={
		"filename": pl.Utf8,
		**{t: pl.Int64 for t in VALID_SVTYPES},
		"illegal": pl.Int64,
		"total_svs": pl.Int64,
	},
)

file_totals.write_csv("sv_count.summary.all.tsv", separator="\t")

# %%
## Count of illegal/missing SVTYPEs that were dropped for each VCFs
illegal_sv_types = pl.DataFrame(
	illegal_sv_types_rows,
	schema={"filename": pl.Utf8, "illegal_svtype": pl.Utf8, "n_dropped": pl.Int64},
)

illegal_sv_types.write_csv("illegal_svtypes.summary.all.tsv", separator="\t")

# %%
## Merge vcfs within platforms
outdir_subplatform = PROJECT_ROOT / "data" / dataset / "sub-platform_merge"
outdir_subplatform.mkdir(parents=True, exist_ok=True)

for (platform,), group_df in annot.group_by("platform"):

	logger.info(f"Processing {group_df.height} VCFs from {platform} platform")

	if "HiC" in platform:
		logger.info("Sikpping HiC VCFs since all SVTYPES are unsupported by OctopuSV")
		continue

	## Merge all SVs i.e PASS & non-PASS
	group_files_all = [DATA_ROOT / "octopusv_correct_all" / filename.replace(".vcf.gz", ".svcf") for filename in group_df["filename"]]
	outpath_all = outdir_subplatform / f"{platform}.merged.all.svcf"
	run(["octopusv", "merge", "-i", *map(str, group_files_all), "-o", str(outpath_all), "--union"])

	## Merge PASS SVs
	group_files_pass = [DATA_ROOT / "octopusv_correct_pass" / filename.replace(".vcf.gz", ".svcf") for filename in group_df["filename"]]
	outpath_pass = outdir_subplatform / f"{platform}.merged.pass.svcf"
	run(["octopusv", "merge", "-i", *map(str, group_files_pass), "-o", str(outpath_pass), "--union"])
	

# %%
## Merge all SVs from subplatform merged SVCF
merged_paths_all = list((DATA_ROOT / "sub-platform_merge").glob("*.all.svcf"))
outpath_merged_all = DATA_ROOT / f"platforms.merged.all.svcf"
run(["octopusv", "merge", "-i", *map(str, merged_paths_all), "-o", str(outpath_merged_all), "--union"])

# %%
## Merge PASS SVs from subplatform merged SVCF
merged_paths_pass = list((DATA_ROOT / "sub-platform_merge").glob("*.pass.svcf"))
outpath_merged_pass = DATA_ROOT / f"platforms.merged.pass.svcf"
run(["octopusv", "merge", "-i", *map(str, merged_paths_pass), "-o", str(outpath_merged_pass), "--union"])


