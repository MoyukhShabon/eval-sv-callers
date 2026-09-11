# %%
import subprocess, tempfile, os
from pathlib import Path
import polars as pl

# %%
project_root = Path("../..").resolve()
annot_root = project_root / "annot"
data_root = project_root / "data"
dataset = "peng24"

# callers to exclude entirely from this merge run
# The callers commented out don't have explicit support in octoupSV documentation
CALLER_BLACKLIST: list[str] = [
	"svision",  # OctopuSV - 1. doesn't support it, 2. chokes on complicated SV such as SVTYPE = INS+tDUP, DUP+INS etc 
	# "NanoSV",
	# "nanovar",
	# "picky"
]


# %%
annot = pl.read_csv(annot_root / dataset / "vcf_annotations.tsv", separator="\t")

if CALLER_BLACKLIST:
	n_before = annot.height
	annot = annot.filter(~pl.col("caller").is_in(CALLER_BLACKLIST))
	print(f"blacklist excluded {n_before - annot.height} row(s) for callers={CALLER_BLACKLIST}")

annot

# %%
fixed_root = data_root / dataset / "vcf_fixed"
svcf_root   = data_root / dataset / "svcf"
merged_root = data_root / dataset / "merged"
svcf_root.mkdir(parents=True, exist_ok=True)
merged_root.mkdir(parents=True, exist_ok=True)


# %%
def run(cmd: list[str]) -> None:
	print("+", " ".join(cmd))
	subprocess.run(cmd, check=True)


groups = (
	annot.select(["platform", "aligner", "sample"])
	.unique()
	.sort(["sample", "platform", "aligner"])
)

# %%
def decompress_and_filter(in_vcf_gz: Path, caller: str, pass_only: bool = False) -> Path:
	"""
	Decompress a caller's gzipped VCF to plain-text VCF via bcftools as
	`octoupsv correct` only supports uncompressed VCF.

	pass_only: keep only FILTER=PASS records (svision has no PASS in its
		FILTER vocabulary. It uses Covered/Uncovered/Clustered, so "Covered"
		is its PASS equivalent).
	pbsv: SVTYPE=cnv records are always dropped, regardless of pass_only, as
	  octopusv's merge step can't classify them and crashes.
	"""
	excludes = []
	if pass_only:
		pass_value = "Covered" if caller == "svision" else "PASS"
		excludes.append(f'FILTER!="{pass_value}"')
	if caller == "pbsv":
		excludes.append('INFO/SVTYPE="cnv" || INFO/SVTYPE="CNV"')

	fd, name = tempfile.mkstemp(suffix=".vcf")
	os.close(fd)
	out_path = Path(name)

	cmd = ["bcftools", "view", str(in_vcf_gz), "-Ov", "-o", str(out_path)]
	if excludes:
		cmd += ["-e", " || ".join(excludes)]

	try:
		subprocess.run(cmd, check=True)
	except BaseException:
		out_path.unlink(missing_ok=True)
		raise
	return out_path

# %%
for platform, aligner, sample in groups.iter_rows():
	grp = annot.filter(
		(pl.col("platform") == platform)
		& (pl.col("aligner") == aligner)
		& (pl.col("sample") == sample)
	)

	tag = f"{platform}_{sample}_{aligner}"
	print(f"\n=== group {tag} ({grp.height} callers) ===")

	svcf_paths: list[tuple[str, Path]] = []

	for row in grp.iter_rows(named=True):
		caller = row["caller"]
		in_vcf_gz = Path(fixed_root / row["filename"])
		print(f"\n=> Processing Caller: {caller} | Path: {in_vcf_gz}")

		if not in_vcf_gz.exists():
			print(f"  [skip] missing: {in_vcf_gz}")
			continue

		# octopusv correct can't read gzip; decompress + drop unsupported SVTYPEs
		tmp_path = decompress_and_filter(in_vcf_gz, caller, pass_only=False)

		out_svcf = svcf_root / f"{row["filename"].split(".")[0]}.svcf"

		try:
			run([
				"octopusv", "correct",
				"-i", str(tmp_path),
				"-o", str(out_svcf),
			])
			svcf_paths.append((caller, out_svcf))
		except subprocess.CalledProcessError as e:
			print(f"  [FAIL] correct failed for {caller}: {e}")
		finally:
			tmp_path.unlink(missing_ok=True)


	if len(svcf_paths) < 2:
		print(f"  [skip merge] only {len(svcf_paths)} caller(s) succeeded, need >=2")
		continue

	merged_out = merged_root / f"{tag}.merged.svcf"
	run([
		"octopusv", "merge",
		"-i", *[str(p) for _, p in svcf_paths],
		"-o", str(merged_out),
		"--union", "--upsetr"
	])
	print(f"  -> {merged_out}  callers={[c for c, _ in svcf_paths]}")


