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
## platform -> merged SVCF + manifest for that platform. The matrix script loops over this file.
OUT_PLATFORM_INDEX = OUTDIR_PLATFORM / "index.tsv"

## tumor.{platform}.{caller}[.{replicate}], e.g. tumor.illumina.delly.IL_T_3, tumor.ont.nanosv
FILENAME_RE = re.compile(
	r"^(?P<tissue>[^.]+)\.(?P<platform>[^.]+)\.(?P<caller>[^.]+)(?:\.(?P<replicate>.+))?$"
)

OUT_MERGED = DATA_ROOT / "platforms.merged.split.svcf"
## label -> platform / file / split. The matrix script reads this, so it never parses label strings.
OUT_MANIFEST = DATA_ROOT / "platforms.merged.split.sources.tsv"

## False reuses existing octopusv_correct_{pass,nonpass} outputs (faster reruns)
RUN_SANITIZE = True

## Platforms left out of the merge (substring match). HiC SVTYPEs are all unsupported by OctopuSV.
SKIP_PLATFORMS = ("HiC",)

## FILTER values that count as PASS; everything else (including ".") is non-PASS.
## VALIDATE_SPLIT checks this against `octopusv correct --filter-pass`.
PASS_FILTERS = frozenset({"PASS"})
VALIDATE_SPLIT = True

## All inputs are one biological sample (HCC1395 tumor), so caller mode throughout.
MERGE_MODE = "caller"

## Contigs to evaluate. Records with ANY endpoint (CHROM, INFO/CHR2, BND ALT mate)
## outside this set are dropped: decoys, chrUn, *_random, *_alt, HLA, chrEBV, viral.
## chrM is kept on purpose.
PRIMARY_CONTIGS = frozenset([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"])

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
## Mate contig + position from a bracketed BND ALT. Greedy contig match splits on the LAST
## colon, so HLA contig names that contain colons (HLA-A*01:01:01:01) still parse correctly.
BND_MATE_RE = re.compile(r"[\[\]]([^\[\]]+):(-?\d+)[\[\]]")
GT_RE = re.compile(r"^[0-9.]+(?:[/|][0-9.]+)*$")
BASES_RE = re.compile(r"[ACGTNacgtn]+")
MISSING_GT = frozenset({".", "./.", ".|.", "./", ".|"})

TRA_DELTA = 500                  # bp; choose from the delta sweep
TRA_STRAND_CONSISTENCY = False

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

##  Drop reason -> summary column name. Order = summary column order.
DROP_REASONS: dict[str, str] = {
	"illegal_svtype": "illegal",        # same column name as before
	"nonprimary":     "n_nonprimary",
	"bad_coord":      "n_bad_coord",
}
MAX_BAD_COORD_WARNINGS = 5   #  per file; bad coords on primary contigs deserve a look

SVLEN_RE = re.compile(r"(?:^|;)SVLEN=([^;]*)")
SVLEN_SUB_RE = re.compile(r"(^|;)SVLEN=[^;]*")
## Intra-chromosomal types where SVLEN = END - POS. DEL left out (sign convention); add it if DELs show 0 too.
FILL_SVLEN_TYPES = frozenset({"INV", "DUP"})

def polish_svcf(path: Path, name: str) -> Counter:
	"""Post-correct cleanup: drop duplicates that differ only in ID, fill SVLEN=0/./missing."""
	fixes, seen = Counter(), set()
	tmp = path.with_name(path.name + ".tmp")
	with open(path) as fin, open(tmp, "w") as fout:
		for line in fin:
			if line.startswith("#"):
				fout.write(line)
				continue
			f = line.rstrip("\r\n").split("\t")
			key = "\t".join(f[:2] + f[3:])          # everything except ID
			if key in seen:
				fixes["dedup_post_correct"] += 1
				continue
			seen.add(key)
			svtype = info_get(SVTYPE_RE, f[7])
			if svtype in FILL_SVLEN_TYPES:
				end, svlen = info_get(END_RE, f[7]), info_get(SVLEN_RE, f[7])
				if end is not None and svlen in (None, ".", "0"):
					n = int(end) - int(f[1])
					f[7] = (f[7] + f";SVLEN={n}" if svlen is None
					        else SVLEN_SUB_RE.sub(rf"\g<1>SVLEN={n}", f[7], count=1))
					fixes[f"filled_svlen_{svtype}"] += 1
			fout.write("\t".join(f) + "\n")
	tmp.replace(path)
	for k, n in sorted(fixes.items()):
		logger.warning(f"[{name}] {k}: {n} record(s)")
	return fixes

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

## 
def contig_class(c: str) -> str:
	"""Coarse class for a non-primary contig, used in drop labels."""
	if c.endswith("_decoy"):  return "decoy"
	if c.endswith("_alt"):    return "alt"
	if c.endswith("_random"): return "unlocalized"
	if c.startswith("chrUn"): return "unplaced"
	if c.startswith("HLA-"):  return "HLA"
	if c == "chrEBV":         return "EBV"
	return "other"            # viral contigs (GRCh38.d1.vd1), non-chr-prefixed names, etc.

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
	Read one VCF(.gz), repair records, drop invalid records, and split by FILTER
	into PASS / non-PASS temp files. Steps per record, in order:

	  1. Trim columns beyond the #CHROM width:
	       - all empty samples            -> trim
	       - non-genotype data (novoBreak) -> trim
	       - genotype data (Delly normal)  -> trim, keeping the declared sample(s);
	         raises at end of file if the dropped sample looks more like the tumor
	     Too few columns -> ValueError.
	  2. Swap ID/REF when REF='.' and ID is bases (novoBreak).
	  3. Drop exact duplicates (compared after 1-2).
	  4. Drop records with any endpoint (CHROM, INFO/CHR2, BND ALT mate) off PRIMARY_CONTIGS.
	  5. Drop records with POS < 1, END < 1, or BND mate position < 1 (backstop; should be
	     ~0 after step 4, so these are logged).
	  6. Drop records with missing/invalid SVTYPE.
	  7. SVTYPE=BND with symbolic ALT -> retype to ALT type if same-chrom DEL/DUP/INV/INS, else drop.
	  8. Intra-chromosomal DEL/DUP/INV with END < POS -> swap.
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
		self._n_bad_coord = 0              # 
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

		##  contig filter, then coordinate backstop, both before any SVTYPE logic
		if not self._check_contigs(fields):
			return None
		if not self._check_coords(fields, lineno):
			return None

		svtype = info_get(SVTYPE_RE, fields[7])
		if svtype not in VALID_SVTYPES:
			self._drop(fields, svtype or "MISSING", "illegal_svtype")
			return None

		if svtype == "BND" and not BREAKEND_ALT_RE.search(fields[4]):
			svtype = self._retype_bnd(fields)
			if svtype is None:
				return None

		if svtype in SWAPPABLE_SVTYPES:
			self._swap_reversed_end(fields, svtype)

		self.svtypes[svtype] += 1
		return fields

	## ---- filters ----
	def _check_contigs(self, fields: list[str]) -> bool:
		contigs = {fields[0]}
		if (chr2 := info_get(CHR2_RE, fields[7])):
			contigs.add(chr2)
		if (m := BND_MATE_RE.search(fields[4])):
			contigs.add(m.group(1))
		off = sorted({contig_class(c) for c in contigs if c not in PRIMARY_CONTIGS})
		if off:
			self._drop(fields, "NONPRIMARY:" + "+".join(off), "nonprimary")
			return False
		return True

	def _check_coords(self, fields: list[str], lineno: int) -> bool:
		bad = []
		if int(fields[1]) < 1:
			bad.append("POS")
		end = info_get(END_RE, fields[7])
		if end is not None and int(end) < 1:
			bad.append("END")
		if (m := BND_MATE_RE.search(fields[4])) and int(m.group(2)) < 1:
			bad.append("MATE")
		if not bad:
			return True

		label = "BAD_COORD:" + "+".join(bad)
		self._drop(fields, label, "bad_coord")
		self._n_bad_coord += 1
		if self._n_bad_coord <= MAX_BAD_COORD_WARNINGS:
			logger.warning(
				f"[{self.name}] line {lineno}: {label} on primary contig(s): "
				f"{fields[0]}:{fields[1]} ALT={fields[4][:60]} INFO={fields[7][:120]}"
			)
		return False

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

		self._drop(fields, f"BND_ALT={alt}", "illegal_svtype")
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

	## takes a reason (key of DROP_REASONS);
	def _drop(self, fields: list[str], label: str, reason: str) -> None:
		assert reason in DROP_REASONS, reason
		self.dropped.append({
			"filename": self.name, "chrom": fields[0], "pos": fields[1],
			"id": fields[2], "reason": reason, "label": label,
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
	dropped: list[dict] = field(default_factory=list)
	fixed: list[dict] = field(default_factory=list)

	def add(self, name: str, res: SanitizeResult) -> None:
		by_reason = Counter(r["reason"] for r in res.dropped)
		by_label = Counter((r["reason"], r["label"]) for r in res.dropped)
		n_valid = sum(res.svtype_counts.values())
		n_dropped = sum(by_reason.values())
		assert res.split_counts["pass"] + res.split_counts["nonpass"] == n_valid

		for kind, desc in FIX_KINDS.items():
			if res.fix_counts[kind]:
				logger.warning(f"[{name}] {desc}: {res.fix_counts[kind]} record(s)")
		for (reason, label), n in sorted(by_label.items()):
			logger.warning(f"[{name}] dropped {n} record(s): {reason} ({label})")

		self.totals.append({
			"filename": name,
			**{t: res.svtype_counts[t] for t in VALID_SVTYPES},
			**{col: by_reason[r] for r, col in DROP_REASONS.items()},
			"total_svs": n_valid + n_dropped,     # after dedup
			"n_pass": res.split_counts["pass"],
			"n_nonpass": res.split_counts["nonpass"],
			**{f"n_{k}": res.fix_counts[k] for k in FIX_KINDS},
		})
		self.filters += [{"filename": name, "filter": f, "n": n} for f, n in res.filter_counts.items()]
		self.dropped += [{"filename": name, "reason": r, "label": lab, "n_dropped": n}
		                 for (r, lab), n in sorted(by_label.items())]
		self.fixed += res.fixed

	def write(self, outdir: Path) -> None:
		tables = {
			"sv_count.summary.all.tsv": (self.totals, {
				"filename": pl.Utf8, **{t: pl.Int64 for t in VALID_SVTYPES},
				**{col: pl.Int64 for col in DROP_REASONS.values()},   
				"total_svs": pl.Int64, "n_pass": pl.Int64, "n_nonpass": pl.Int64,
				**{f"n_{k}": pl.Int64 for k in FIX_KINDS},
			}),
			"filter_values.summary.tsv": (self.filters, {
				"filename": pl.Utf8, "filter": pl.Utf8, "n": pl.Int64,
			}),
			"dropped_records.summary.all.tsv": (self.dropped, {
				"filename": pl.Utf8, "reason": pl.Utf8, "label": pl.Utf8, "n_dropped": pl.Int64,
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
	if len(inputs) != len(labels):
		raise ValueError(f"{len(inputs)} inputs but {len(labels)} labels")
	if len(set(labels)) != len(labels):
		raise ValueError(f"duplicate merge labels: {labels}")
	bad = [lab for lab in labels if re.search(r"[,;=\s]", lab)]
	if bad:
		raise ValueError(f"merge labels must not contain ',', ';', '=' or whitespace: {bad}")
	missing = [p for p in inputs if not p.exists()]
	if missing:
		raise FileNotFoundError(f"merge inputs missing: {missing}")
	names_flag = "--caller-names" if MERGE_MODE == "caller" else "--sample-names"
	dst.unlink(missing_ok=True)
	## All inputs positional (no -i): OctopuSV forbids mixing input styles with --caller-names,
	## and -i only takes one file, so `-i A B C` was read as -i A plus positional B, C.
	cmd = ["octopusv", "merge", *map(str, inputs), "-o", str(dst), "--union",
	       "--mode", MERGE_MODE, names_flag, ",".join(labels),
	       "--tra-delta", str(TRA_DELTA)]
	if not TRA_STRAND_CONSISTENCY:
		cmd.append("--no-tra-strand-consistency")
	run(cmd)

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
	polish_svcf(dst, vcf.stem)
	truth_svcfs.append(dst)

# %%
## ---------------- sanitize, split, correct ----------------
if annot["filename"].is_duplicated().any():
	raise ValueError("duplicate filenames in annotation table; merge labels would collide")

if RUN_SANITIZE:
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
				polish_svcf(dst, f"{name}.{split}")
		finally:
			res.cleanup()

	summaries.write(SUMMARY_DIR)
else:
	logger.warning(
		f"RUN_SANITIZE=False: reusing corrected SVCFs in {list(map(str, OUTDIR_CORRECT.values()))}. "
		"Make sure they come from the current sanitizer and annotation table."
	)

# %%
## ---------------- source manifest ----------------
MANIFEST_SCHEMA = {
	"label": pl.Utf8,      # exact string OctopuSV writes into INFO/SOURCES
	"kind": pl.Utf8,       # "call" | "truth"
	"platform": pl.Utf8,   # matrix column at level="platform"
	"caller": pl.Utf8,     # matrix column at level="caller" (ORs over replicates)
	"replicate": pl.Utf8,  # e.g. IL_T_3, A1; null for single-run callers (ont/pacbio)
	"name": pl.Utf8,       # matrix column at level="file"
	"split": pl.Utf8,      # "pass" | "nonpass" | null (truth)
	"path": pl.Utf8,
}

def file_meta(rec: dict) -> tuple[str, str, str | None]:
	"""(name, caller, replicate) for one annotation row. Uses annot's caller column if there is one."""
	name = rec["filename"].removesuffix(".vcf.gz")
	m = FILENAME_RE.match(name)
	if m is None:
		raise ValueError(f"cannot parse caller/replicate from filename: {name}")
	if m["platform"].lower() != str(rec["platform"]).lower():
		logger.warning(f"[{name}] filename platform '{m['platform']}' != annot platform '{rec['platform']}'")
	caller = rec.get("caller") or m["caller"]
	return name, caller, m["replicate"]

def build_manifest(annot: pl.DataFrame, truth_svcfs: list[Path]) -> pl.DataFrame:
	rows = []
	for rec in annot.iter_rows(named=True):
		platform = rec["platform"]
		if any(s in platform for s in SKIP_PLATFORMS):
			continue
		name, caller, replicate = file_meta(rec)
		for split in SPLITS:
			path = OUTDIR_CORRECT[split] / f"{name}.svcf"
			if not path.exists():      # e.g. a caller that emits PASS only
				continue
			rows.append({"label": f"{name}.{split}", "kind": "call", "platform": platform,
			             "caller": caller, "replicate": replicate, "name": name,
			             "split": split, "path": str(path)})

	for path in truth_svcfs:
		lab = path.stem                # e.g. SEQC2_Consensus_SVs -> column seqc2_consensus_svs
		rows.append({"label": lab, "kind": "truth", "platform": lab, "caller": lab,
		             "replicate": None, "name": lab, "split": None, "path": str(path)})

	manifest = pl.DataFrame(rows, schema=MANIFEST_SCHEMA)
	if manifest["label"].is_duplicated().any():
		raise ValueError("duplicate labels in source manifest")

	logger.info(
		"Merge inputs per platform/caller/split:\n"
		+ str(manifest.group_by("kind", "platform", "caller", "split", maintain_order=True).len())
	)
	return manifest

manifest = build_manifest(annot, truth_svcfs)

# %%
## ---------------- flat merge: every file x split + truth, one clustering step ----------------
## Remove the old manifest first so a failed merge can't leave a stale one next to an old SVCF
OUT_MANIFEST.unlink(missing_ok=True)
octopusv_merge([Path(p) for p in manifest["path"]], manifest["label"].to_list(), OUT_MERGED)
manifest.write_csv(OUT_MANIFEST, separator="\t")
logger.info(f"Wrote {OUT_MERGED} ({count_records(OUT_MERGED)} records) and {OUT_MANIFEST}")

# %%
## ---------------- intra-platform merges: one platform's files x splits + truth sets ----------------
## Same PASS/non-PASS rule as the flat merge. Each platform is clustered separately, so its
## records are not shaped by other platforms' calls.
OUTDIR_PLATFORM.mkdir(parents=True, exist_ok=True)
OUT_PLATFORM_INDEX.unlink(missing_ok=True)   # a failed run can't leave a stale index behind

calls = manifest.filter(pl.col("kind") == "call")
truth = manifest.filter(pl.col("kind") == "truth")

index_rows = []
for (platform,), pm in calls.group_by("platform", maintain_order=True):
	slug = re.sub(r"[^\w.-]", "_", platform)
	dst = OUTDIR_PLATFORM / f"{slug}.merged.split.svcf"
	dst_manifest = OUTDIR_PLATFORM / f"{slug}.merged.split.sources.tsv"
	dst_manifest.unlink(missing_ok=True)

	callers = pm["caller"].unique(maintain_order=True).to_list()
	if len(callers) == 1:
		logger.info(f"[{platform}]: only one caller ({callers[0]}); matrix compares replicates/truth only")
	if set(c.lower() for c in callers) & set(t.lower() for t in truth["caller"]):
		raise ValueError(f"[{platform}]: caller name collides with a truth-set label")

	sub = pl.concat([pm, truth])
	logger.info(f"[{platform}]: merging {pm['name'].n_unique()} files ({len(callers)} callers) "
	            f"x splits = {pm.height} inputs + {truth.height} truth sets")
	octopusv_merge([Path(p) for p in sub["path"]], sub["label"].to_list(), dst)
	sub.write_csv(dst_manifest, separator="\t")

	index_rows.append({
		"platform": platform, "slug": slug,
		"n_files": pm["name"].n_unique(), "n_callers": len(callers),
		"callers": ",".join(callers), "n_records": count_records(dst),
		"svcf": str(dst), "manifest": str(dst_manifest),
	})

# %%
platform_index = pl.DataFrame(index_rows)
platform_index.write_csv(OUT_PLATFORM_INDEX, separator="\t")
logger.info(f"Wrote {OUT_PLATFORM_INDEX}:\n{platform_index.drop('svcf', 'manifest')}")