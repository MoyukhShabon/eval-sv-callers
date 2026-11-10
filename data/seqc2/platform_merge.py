# %%
import gzip
import logging
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

# %%
logging.basicConfig(
	level=logging.INFO,
	format="\n==[%(levelname)s | %(asctime)s]: %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	force=True,
)
logger = logging.getLogger("platform_merge")

# %%
## ---------------- configuration ----------------
PROJECT_ROOT = Path("../..").resolve()
DATASET = "seqc2"
DATA_ROOT = PROJECT_ROOT / "data" / DATASET
VCF_DIR = DATA_ROOT / "vcf"
ANNOT_TSV = PROJECT_ROOT / "annot" / DATASET / "vcf_annotations.tsv"
SUMMARY_DIR = Path(".")

TRUTH_VCFS = [
	DATA_ROOT / "SEQC2_Consensus_SVs.vcf",
	DATA_ROOT / "SEQC2_Validated_SVs.vcf",
	DATA_ROOT / "SEQC2_PCR_SVs.vcf",
]

SPLITS = ("pass", "nonpass")
OUTDIR_CORRECT = {s: DATA_ROOT / f"octopusv_correct_{s}" for s in SPLITS}
OUTDIR_PLATFORM = DATA_ROOT / "intra-platform_merge"
OUT_MERGED = DATA_ROOT / "platforms.merged.split.svcf"

## FILTER values that count as PASS; everything else (including ".") is non-PASS.
## VALIDATE_SPLIT checks this against `octopusv correct --filter-pass`.
PASS_FILTERS = frozenset({"PASS"})
VALIDATE_SPLIT = True

## All inputs are one biological sample (HCC1395 tumor), so caller mode throughout.
MERGE_MODE = "caller"

# %%
## ---------------- VCF grammar ----------------
## Tuple, not set: the order sets the summary column order, and set order of str changes between runs
VALID_SVTYPES = ("DEL", "DUP", "INS", "INV", "BND", "TRA")

## Intra-chromosomal types where swapping POS/END describes the same interval.
## INS excluded (would move the insertion point); TRA/BND have END on CHR2.
SWAPPABLE_SVTYPES = frozenset({"DEL", "DUP", "INV"})
RETYPEABLE_FROM_BND = frozenset({"DEL", "DUP", "INV", "INS"})

## (^|;) anchors stop these from matching CIEND= / SVEND= etc.
SVTYPE_RE = re.compile(r"(?:^|;)SVTYPE=([^;]+)")
SVTYPE_SUB_RE = re.compile(r"(^|;)SVTYPE=[^;]+")
END_RE = re.compile(r"(?:^|;)END=(-?\d+)")
END_SUB_RE = re.compile(r"(^|;)END=-?\d+")
CHR2_RE = re.compile(r"(?:^|;)CHR2=([^;]+)")

SYMBOLIC_ALT_RE = re.compile(r"^<([A-Za-z]+)(?::[^>]*)?>$")           # <INV>, <DUP:TANDEM>
BREAKEND_ALT_RE = re.compile(r"[\[\]]|^\.[A-Za-z]+$|^[A-Za-z]+\.$")   # N[chr:pos[, ]chr:pos]N, N., .N
GT_RE = re.compile(r"^[0-9.]+(?:[/|][0-9.]+)*$")
BASES_RE = re.compile(r"[ACGTNacgtn]+")
MISSING_GT = frozenset({".", "./.", ".|.", "./", ".|"})

## Repair kind -> log description. Order = summary column order.
FIX_KINDS: dict[str, str] = {
	"trimmed_empty_cols":  "trimmed empty undeclared sample column(s)",
	"trimmed_annot_cols":  "trimmed undeclared non-sample column(s) (e.g. novoBreak annotations)",
	"trimmed_sample_cols": "trimmed undeclared sample column(s) WITH data (e.g. Delly matched normal)",
	"swapped_id_ref":      "swapped ID/REF (ID=<bases>, REF=.)",
	"dedup":               "dropped exact duplicate record(s)",
	"retyped_bnd":         "retyped SVTYPE=BND to match symbolic ALT",
	"swapped_end":         "swapped POS/END with END < POS",
}

# %%
## ---------------- helpers ----------------
def run(cmd: list[str]) -> None:
	logger.info("=> " + " ".join(cmd))
	subprocess.run(cmd, check=True)

def count_records(path: Path) -> int:
	with open(path) as f:
		return sum(1 for line in f if not line.startswith("#"))

def open_text(path: Path):
	return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)

def info_get(regex: re.Pattern, info: str) -> str | None:
	m = regex.search(info)
	return m.group(1) if m else None

def is_empty_sample(col: str) -> bool:
	"""True if a sample column has no genotype (missing GT or blank)."""
	col = col.strip()
	return col in ("", ".") or col.split(":", 1)[0] in MISSING_GT

def gt_status(fmt: str, sample: str) -> bool | None:
	"""True = non-ref GT, False = ref/missing GT, None = not a genotype."""
	if fmt.split(":", 1)[0] != "GT":
		return None
	gt = sample.split(":", 1)[0]
	if not GT_RE.match(gt):
		return None
	return any(a not in ("0", ".") for a in re.split(r"[/|]", gt))

# %%
## ---------------- sanitizer ----------------
@dataclass
class SanitizeResult:
	split_paths: dict[str, Path]          # plain-text VCFs, full header in each
	split_counts: Counter
	svtype_counts: Counter
	filter_counts: Counter
	fix_counts: Counter
	dropped: list[dict]
	fixed: list[dict]

	def cleanup(self) -> None:
		for p in self.split_paths.values():
			p.unlink(missing_ok=True)


class VcfSanitizer:
	"""
	Read one VCF(.gz), repair records, drop invalid SVTYPEs, and split by FILTER
	into PASS / non-PASS temp files. Steps per record, in order:

	  1. Trim columns beyond the #CHROM width:
	       - all empty samples            -> trim
	       - non-genotype data (novoBreak) -> trim
	       - genotype data (Delly normal)  -> trim, keeping the declared sample(s);
	         raises at end of file if the dropped sample looks more like the tumor
	     Too few columns -> ValueError.
	  2. Swap ID/REF when REF='.' and ID is bases (novoBreak).
	  3. Drop exact duplicates (compared after 1-2).
	  4. Drop records with missing/invalid SVTYPE.
	  5. SVTYPE=BND with symbolic ALT -> retype to ALT type if same-chrom DEL/DUP/INV/INS, else drop.
	  6. Intra-chromosomal DEL/DUP/INV with END < POS -> swap.
	"""

	def __init__(self, path: Path):
		self.path = path
		self.name = path.name
		self.n_cols: int | None = None
		self.svtypes: Counter = Counter()
		self.filters: Counter = Counter()
		self.splits: Counter = Counter()
		self.fixes: Counter = Counter()
		self.dropped: list[dict] = []
		self.fixed: list[dict] = []
		self._nonref_kept = 0
		self._nonref_dropped = 0
		self._seen: set[str] = set()

	## ---- driver ----
	def run(self) -> SanitizeResult:
		paths: dict[str, Path] = {}
		for split in SPLITS:
			fd, tmp = tempfile.mkstemp(suffix=f".{split}.vcf")
			os.close(fd)
			paths[split] = Path(tmp)
		result = SanitizeResult(paths, self.splits, self.svtypes, self.filters,
		                        self.fixes, self.dropped, self.fixed)
		try:
			with open_text(self.path) as fin, \
			     open(paths["pass"], "w") as f_pass, \
			     open(paths["nonpass"], "w") as f_nonpass:
				outs = {"pass": f_pass, "nonpass": f_nonpass}

				for lineno, line in enumerate(fin, start=1):
					if line.startswith("#"):
						if line.startswith("#CHROM"):
							self.n_cols = len(line.rstrip("\r\n").split("\t"))
						for f in outs.values():
							f.write(line)
						continue
					if not line.strip():
						continue
					if self.n_cols is None:
						raise ValueError(f"{self.name}:{lineno}: data record before #CHROM header line")

					fields = self._process(line.rstrip("\r\n").split("\t"), lineno)
					if fields is None:
						continue

					filt = fields[6]
					split = "pass" if filt in PASS_FILTERS else "nonpass"
					self.filters[filt] += 1
					self.splits[split] += 1
					outs[split].write("\t".join(fields) + "\n")

			self._check_sample_orientation()
		except BaseException:
			result.cleanup()
			raise
		return result

	def _process(self, fields: list[str], lineno: int) -> list[str] | None:
		fields = self._trim_columns(fields, lineno)
		self._fix_id_ref(fields)

		key = "\t".join(fields)
		if key in self._seen:
			self.fixes["dedup"] += 1
			return None
		self._seen.add(key)

		svtype = info_get(SVTYPE_RE, fields[7])
		if svtype not in VALID_SVTYPES:
			self._drop(fields, svtype or "MISSING")
			return None

		if svtype == "BND" and not BREAKEND_ALT_RE.search(fields[4]):
			svtype = self._retype_bnd(fields)
			if svtype is None:
				return None

		if svtype in SWAPPABLE_SVTYPES:
			self._swap_reversed_end(fields, svtype)

		self.svtypes[svtype] += 1
		return fields

	## ---- repairs ----
	def _trim_columns(self, fields: list[str], lineno: int) -> list[str]:
		n = self.n_cols
		if len(fields) < n:
			raise ValueError(f"{self.name}:{lineno}: record has {len(fields)} columns but #CHROM declares {n}")
		if len(fields) == n:
			return fields

		extra = fields[n:]
		if all(is_empty_sample(c) for c in extra):
			kind = "trimmed_empty_cols"
		elif n > 9 and gt_status(fields[8], extra[0]) is not None:
			kind = "trimmed_sample_cols"
			self._nonref_kept += bool(gt_status(fields[8], fields[9]))
			self._nonref_dropped += bool(gt_status(fields[8], extra[0]))
		else:
			kind = "trimmed_annot_cols"

		self.fixes[kind] += 1
		if self.fixes[kind] == 1:
			logger.warning(
				f"[{self.name}] line {lineno}: {len(extra)} undeclared column(s) ({kind}); "
				f"first dropped value: {extra[0][:80]}"
			)
		return fields[:n]

	def _fix_id_ref(self, fields: list[str]) -> None:
		if fields[3] == "." and BASES_RE.fullmatch(fields[2]):
			fields[2], fields[3] = ".", fields[2]
			self.fixes["swapped_id_ref"] += 1

	def _retype_bnd(self, fields: list[str]) -> str | None:
		chrom, pos, alt, info = fields[0], fields[1], fields[4], fields[7]
		m = SYMBOLIC_ALT_RE.match(alt)
		alt_type = m.group(1) if m else None
		same_chrom = (info_get(CHR2_RE, info) or chrom) == chrom
		end = info_get(END_RE, info)

		if alt_type in RETYPEABLE_FROM_BND and same_chrom and (end is not None or alt_type == "INS"):
			fields[7] = SVTYPE_SUB_RE.sub(rf"\g<1>SVTYPE={alt_type}", info, count=1)
			self._record_fix(fields, int(pos), int(end) if end else None, alt_type,
			                 f"retyped_BND_to_{alt_type}", "retyped_bnd")
			return alt_type

		self._drop(fields, f"BND_ALT={alt}")
		return None

	def _swap_reversed_end(self, fields: list[str], svtype: str) -> None:
		chrom, info = fields[0], fields[7]
		end = info_get(END_RE, info)
		if end is None or (info_get(CHR2_RE, info) or chrom) != chrom:
			return
		pos_i, end_i = int(fields[1]), int(end)
		if end_i >= pos_i:
			return
		fields[1] = str(end_i)
		fields[7] = END_SUB_RE.sub(rf"\g<1>END={pos_i}", info, count=1)
		self._record_fix(fields, pos_i, end_i, svtype, "swapped_pos_end", "swapped_end")

	## ---- bookkeeping ----
	def _record_fix(self, fields, orig_pos, orig_end, svtype, fix, kind) -> None:
		self.fixes[kind] += 1
		self.fixed.append({
			"filename": self.name, "chrom": fields[0], "orig_pos": orig_pos, "orig_end": orig_end,
			"id": fields[2], "svtype": svtype, "filter": fields[6], "fix": fix,
		})

	def _drop(self, fields: list[str], label: str) -> None:
		self.dropped.append({
			"filename": self.name, "chrom": fields[0], "pos": fields[1],
			"id": fields[2], "illegal_svtype": label,
		})

	def _check_sample_orientation(self) -> None:
		if self.fixes["trimmed_sample_cols"] and self._nonref_dropped > self._nonref_kept:
			raise ValueError(
				f"{self.name}: the dropped undeclared sample has more non-ref GTs "
				f"({self._nonref_dropped}) than the kept one ({self._nonref_kept}). Sample order is "
				f"probably NORMAL,TUMOR, so trimming would keep the normal. Handle this file explicitly."
			)

# %%
## ---------------- summaries ----------------
@dataclass
class Summaries:
	totals: list[dict] = field(default_factory=list)
	filters: list[dict] = field(default_factory=list)
	illegal: list[dict] = field(default_factory=list)
	fixed: list[dict] = field(default_factory=list)

	def add(self, name: str, res: SanitizeResult) -> None:
		illegal = Counter(r["illegal_svtype"] for r in res.dropped)
		n_valid = sum(res.svtype_counts.values())
		n_illegal = sum(illegal.values())
		assert res.split_counts["pass"] + res.split_counts["nonpass"] == n_valid

		for kind, desc in FIX_KINDS.items():
			if res.fix_counts[kind]:
				logger.warning(f"[{name}] {desc}: {res.fix_counts[kind]} record(s)")
		for label, n in illegal.items():
			logger.warning(f"[{name}] dropped {n} record(s) with SVTYPE={label}")

		self.totals.append({
			"filename": name,
			**{t: res.svtype_counts[t] for t in VALID_SVTYPES},
			"illegal": n_illegal,
			"total_svs": n_valid + n_illegal,     # after dedup
			"n_pass": res.split_counts["pass"],
			"n_nonpass": res.split_counts["nonpass"],
			**{f"n_{k}": res.fix_counts[k] for k in FIX_KINDS},
		})
		self.filters += [{"filename": name, "filter": f, "n": n} for f, n in res.filter_counts.items()]
		self.illegal += [{"filename": name, "illegal_svtype": t, "n_dropped": n} for t, n in illegal.items()]
		self.fixed += res.fixed

	def write(self, outdir: Path) -> None:
		tables = {
			"sv_count.summary.all.tsv": (self.totals, {
				"filename": pl.Utf8, **{t: pl.Int64 for t in VALID_SVTYPES},
				"illegal": pl.Int64, "total_svs": pl.Int64, "n_pass": pl.Int64, "n_nonpass": pl.Int64,
				**{f"n_{k}": pl.Int64 for k in FIX_KINDS},
			}),
			"filter_values.summary.tsv": (self.filters, {
				"filename": pl.Utf8, "filter": pl.Utf8, "n": pl.Int64,
			}),
			"illegal_svtypes.summary.all.tsv": (self.illegal, {
				"filename": pl.Utf8, "illegal_svtype": pl.Utf8, "n_dropped": pl.Int64,
			}),
			"fixed_records.summary.all.tsv": (self.fixed, {
				"filename": pl.Utf8, "chrom": pl.Utf8, "orig_pos": pl.Int64, "orig_end": pl.Int64,
				"id": pl.Utf8, "svtype": pl.Utf8, "filter": pl.Utf8, "fix": pl.Utf8,
			}),
		}
		for fname, (rows, schema) in tables.items():
			pl.DataFrame(rows, schema=schema).write_csv(outdir / fname, separator="\t")

# %%
## ---------------- OctopuSV wrappers ----------------
def octopusv_correct(src: Path, dst: Path, *, split: str, name: str, filter_counts: Counter) -> None:
	dst.unlink(missing_ok=True)
	run(["octopusv", "correct", "-i", str(src), "-o", str(dst)])
	if not VALIDATE_SPLIT or split not in SPLITS:
		return

	## pass split -> --filter-pass keeps everything; nonpass split -> keeps nothing
	fd, chk = tempfile.mkstemp(suffix=".svcf")
	os.close(fd)
	chk = Path(chk)
	try:
		run(["octopusv", "correct", "-i", str(src), "-o", str(chk), "--filter-pass"])
		expected = count_records(dst) if split == "pass" else 0
		got = count_records(chk)
		if got != expected:
			logger.warning(
				f"[{name}] {split}: --filter-pass kept {got}, expected {expected}. "
				f"PASS_FILTERS disagrees with OctopuSV; FILTER values: {dict(filter_counts)}"
			)
	finally:
		chk.unlink(missing_ok=True)


def octopusv_merge(inputs: list[Path], labels: list[str], dst: Path) -> None:
	if len(set(labels)) != len(labels):
		raise ValueError(f"duplicate merge labels: {labels}")
	if any("," in lab for lab in labels):
		raise ValueError(f"merge labels must not contain commas: {labels}")
	names_flag = "--caller-names" if MERGE_MODE == "caller" else "--sample-names"
	dst.unlink(missing_ok=True)
	run(["octopusv", "merge", "-i", *map(str, inputs), "-o", str(dst), "--union",
	     "--mode", MERGE_MODE, names_flag, ",".join(labels)])

# %%
## ---------------- inputs ----------------
## SEQC2 already removed germline SVs from tumor VCFs
annot = (
	pl.read_csv(ANNOT_TSV, separator="\t")
	.drop("site", "tissue_code", "replicate")
	.filter(pl.col("tissue") == "tumor")
)

# %%
## ---------------- truth sets ----------------
truth_svcfs: list[Path] = []
for vcf in TRUTH_VCFS:
	dst = vcf.with_suffix(".svcf")
	octopusv_correct(vcf, dst, split="truth", name=vcf.stem, filter_counts=Counter())
	truth_svcfs.append(dst)

# %%
## ---------------- sanitize, split, correct ----------------
for d in OUTDIR_CORRECT.values():
	d.mkdir(parents=True, exist_ok=True)

summaries = Summaries()
for filename in annot["filename"]:
	name = filename.removesuffix(".vcf.gz")
	logger.info(f"Sanitizing, splitting & correcting: {name}")

	res = VcfSanitizer(VCF_DIR / filename).run()
	try:
		summaries.add(name, res)
		for split in SPLITS:
			dst = OUTDIR_CORRECT[split] / f"{name}.svcf"
			if res.split_counts[split] == 0:
				dst.unlink(missing_ok=True)   # don't let a stale file from a previous run sneak in
				logger.info(f"[{name}]: 0 {split} records, skipping correct")
				continue
			octopusv_correct(res.split_paths[split], dst, split=split, name=name,
			                 filter_counts=res.filter_counts)
	finally:
		res.cleanup()

summaries.write(SUMMARY_DIR)

# %%
## ---------------- intra-platform merge (per split) ----------------
OUTDIR_PLATFORM.mkdir(parents=True, exist_ok=True)
platform_merged: dict[str, list[Path]] = {s: [] for s in SPLITS}

for (platform,), group_df in annot.group_by("platform", maintain_order=True):
	logger.info(f"Processing {group_df.height} VCFs from {platform} platform")
	if "HiC" in platform:
		logger.info("Skipping HiC VCFs since all SVTYPEs are unsupported by OctopuSV")
		continue

	for split in SPLITS:
		dst = OUTDIR_PLATFORM / f"{platform}.merged.{split}.svcf"
		names = [fn.removesuffix(".vcf.gz") for fn in group_df["filename"]]
		## A replicate may have no records for this split (e.g. a caller that emits PASS only)
		pairs = [(p, n) for n in names if (p := OUTDIR_CORRECT[split] / f"{n}.svcf").exists()]
		if not pairs:
			dst.unlink(missing_ok=True)
			logger.info(f"[{platform}]: no {split} inputs, skipping")
			continue

		inputs, labels = map(list, zip(*pairs))
		octopusv_merge(inputs, labels, dst)
		platform_merged[split].append(dst)

# %%
## ---------------- inter-platform merge (PASS + non-PASS + truth) ----------------
## Use files produced in this run, not a glob, so stale outputs are never picked up
final_inputs = platform_merged["pass"] + platform_merged["nonpass"] + truth_svcfs
final_labels = [p.name.removesuffix(".svcf") for p in final_inputs]
octopusv_merge(final_inputs, final_labels, OUT_MERGED)
