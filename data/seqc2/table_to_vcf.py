#!/usr/bin/env python
"""Convert the SEQC2 consensus SV table (CSV) to a sites-only VCF 4.2."""

import datetime
import re
import sys
from pathlib import Path

import polars as pl
import pysam

# ============================================================== CONFIG
PROJECT_ROOT = Path("../..").resolve()
dataset = "seqc2"


# ====================================================================

# PLATFORM_COLS = {
#     "Called by Pacbio": "PacBio",
#     "Called by Nanopore": "Nanopore",
#     "Called by HiC": "HiC",
#     "Called by Illumina": "Illumina",
#     "Called by 10X": "10X",
# }
SYMBOLIC = {"DEL", "DUP", "INS", "INV"}
VALID_STRANDS = {"++", "+-", "-+", "--"}

HEADER_LINES = [
    "##fileformat=VCFv4.2",
    '##FILTER=<ID=PASS,Description="All filters passed">',
    '##ALT=<ID=DEL,Description="Deletion">',
    '##ALT=<ID=DUP,Description="Duplication">',
    '##ALT=<ID=INS,Description="Insertion">',
    '##ALT=<ID=INV,Description="Inversion">',
    '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">',
    '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">',
    '##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="SV length (negative for deletions)">',
    '##INFO=<ID=CHR2,Number=1,Type=String,Description="Chromosome of the mate breakpoint">',
    '##INFO=<ID=MATEID,Number=.,Type=String,Description="ID of the mate breakend">',
    '##INFO=<ID=STRANDS,Number=1,Type=String,Description="Breakpoint orientation parsed from SV_id (e.g. +-)">',
    '##INFO=<ID=SVID,Number=1,Type=String,Description="Original SV_id from the consensus table">',
    # '##INFO=<ID=NREPS,Number=1,Type=Integer,Description="Number of replicates in which the SV was called">',
    # '##INFO=<ID=PLATFORMS,Number=.,Type=String,Description="Platforms that called the SV">',
    # '##INFO=<ID=NPLAT,Number=1,Type=Integer,Description="Number of platforms that called the SV">',
    # '##INFO=<ID=VALIDATION,Number=.,Type=String,Description="Orthogonal validation method(s)">',
    # '##INFO=<ID=VALIDATED,Number=0,Type=Flag,Description="SV was orthogonally validated">',
]


# ----------------------------------------------------------------- helpers
def to_bool(x):
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "t", "1", "yes"}


def clean_token(s):
    """Make a string safe for an INFO value (no spaces, ';', '=')."""
    return re.sub(r"[\s;=]+", "_", s.strip())


def chrom_sort_key(name):
    """Fallback natural chromosome ordering: chr1..chr22, chrX, chrY, chrM, others."""
    c = name[3:] if name.lower().startswith("chr") else name
    if c.isdigit():
        return (0, int(c), "")
    order = {"X": 1, "Y": 2, "M": 3, "MT": 3}
    if c.upper() in order:
        return (1, order[c.upper()], "")
    return (2, 0, c)


def bnd_alt(ref, self_strand, other_strand, mchrom, mpos):
    """Build a VCF BND ALT allele.
    '+' on this end = sequence to the left of the breakpoint is retained.
    ++ -> N]p]   +- -> N[p[   -+ -> ]p]N   -- -> [p[N
    """
    br = "]" if other_strand == "+" else "["
    p = f"{mchrom}:{mpos}"
    return f"{ref}{br}{p}{br}" if self_strand == "+" else f"{br}{p}{br}{ref}"


def info_str(d):
    parts = []
    for k, v in d.items():
        if v is None or v is False:
            continue
        parts.append(k if v is True else f"{k}={v}")
    return ";".join(parts) if parts else "."


# ----------------------------------------------------------------- records
def build_records(row, fasta):
    """Return a list of (chrom, pos, id, ref, alt, info) tuples for one table row."""
    sv_id = row["SV_id"]
    chrom1, chrom2 = row["Chrom1"], row["Chrom2"]
    pos1, pos2 = int(row["Pos1"]), int(row["Pos2"])
    size = int(row["size"] or 0)
    svtype = str(row["SV_type"]).strip().upper()

    def ref_base(chrom, pos):
        if fasta is None:
            return "N"
        return fasta.fetch(chrom, pos - 1, pos).upper() or "N"

    # shared annotations
    # plats = [name for col, name in PLATFORM_COLS.items() if int(row.get(col) or 0) == 1]
    # validated = to_bool(row.get("is_validated"))
    # vtype = row.get("Validation_type")
    # vlist = None
    # if vtype not in (None, ""):
    #     vlist = ",".join(clean_token(v) for v in re.split(r"[,;|]", str(vtype)) if v.strip())

    common = {
        "SVID": sv_id,
        # "NREPS": row.get("nreps"),
        # "PLATFORMS": ",".join(plats) or None,
        # "NPLAT": len(plats),
        # "VALIDATION": vlist,
        # "VALIDATED": validated,
    }

    strands = sv_id.rsplit("_", 1)[-1]
    if strands not in VALID_STRANDS:
        strands = None

    # ---- breakends (translocations / any inter-chromosomal event)
    if svtype in {"TRA", "BND"} or chrom1 != chrom2:
        if strands is None:
            raise ValueError(f"cannot parse orientation from SV_id '{sv_id}'")
        s1, s2 = strands[0], strands[1]
        id1, id2 = f"{sv_id}_1", f"{sv_id}_2"
        r1, r2 = ref_base(chrom1, pos1), ref_base(chrom2, pos2)
        i1 = {"SVTYPE": "BND", "CHR2": chrom2, "MATEID": id2, "STRANDS": strands, **common}
        i2 = {"SVTYPE": "BND", "CHR2": chrom1, "MATEID": id1, "STRANDS": strands, **common}
        return [
            (chrom1, pos1, id1, r1, bnd_alt(r1, s1, s2, chrom2, pos2), info_str(i1)),
            (chrom2, pos2, id2, r2, bnd_alt(r2, s2, s1, chrom1, pos1), info_str(i2)),
        ]

    # ---- symbolic intra-chromosomal alleles
    if svtype not in SYMBOLIC:
        print(f"WARNING: unknown SV_type '{svtype}' for {sv_id}; using <{svtype}>", file=sys.stderr)

    if svtype != "INS" and pos2 < pos1:
        pos1, pos2 = pos2, pos1

    if svtype == "INS":
        end, svlen = pos1, abs(size)
    elif svtype == "DEL":
        end = pos2
        svlen = -abs(size) if size else -(pos2 - pos1)
    else:  # DUP, INV, other
        end = pos2
        svlen = abs(size) if size else pos2 - pos1

    info = {"SVTYPE": svtype, "END": end, "SVLEN": svlen, "STRANDS": strands, **common}
    return [(chrom1, pos1, sv_id, ref_base(chrom1, pos1), f"<{svtype}>", info_str(info))]


# -------------------------------------------------------------------- main
def convert(input_csv: Path, genome_fai: Path, genome_fasta: Path = None, compress_bgzip = True):
    output_vcf = input_csv.with_suffix(".vcf")

    # ---- read table; strip stray whitespace in headers (e.g. ' # Replicates called')
    df = pl.read_csv(input_csv, infer_schema_length=10000)
    df = df.rename({c: c.strip() for c in df.columns})
    df = df.rename({
        "SV_Size or breakpoints distance": "size",
        # "# Replicates called": "nreps",
    })

    # drop rows with missing coordinates (e.g. from a full join)
    n0 = df.height
    df = df.drop_nulls(["SV_id", "Chrom1", "Pos1", "Chrom2", "Pos2", "SV_type"])
    if df.height < n0:
        print(f"WARNING: dropped {n0 - df.height} rows with missing coordinates", file=sys.stderr)

    if df["SV_id"].n_unique() != df.height:
        dup = df.filter(pl.col("SV_id").is_duplicated())["SV_id"].unique().to_list()[:5]
        sys.exit(f"ERROR: SV_id is not unique, e.g. {dup}")

    # ---- reference / contigs
    fasta = pysam.FastaFile(str(genome_fasta)) if genome_fasta else None
    contigs = []  # (name, length or None)
    if fasta:
        contigs = list(zip(fasta.references, fasta.lengths))
    elif genome_fai:
        with open(genome_fai) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                contigs.append((f[0], int(f[1])))

    known = {c for c, _ in contigs}
    used = set(df["Chrom1"].unique()) | set(df["Chrom2"].unique())
    missing = sorted(used - known, key=chrom_sort_key)
    if contigs and missing:
        print(f"WARNING: contigs not in reference: {missing}", file=sys.stderr)
    if not contigs:
        missing = sorted(used, key=chrom_sort_key)
    contigs += [(c, None) for c in missing]
    order = {c: i for i, (c, _) in enumerate(contigs)}

    # ---- build records
    records, skipped = [], 0
    for row in df.iter_rows(named=True):
        try:
            records.extend(build_records(row, fasta))
        except ValueError as e:
            print(f"WARNING: skipping row: {e}", file=sys.stderr)
            skipped += 1
    records.sort(key=lambda r: (order[r[0]], r[1], r[2]))

    # ---- write
    output_vcf.parent.mkdir(parents=True, exist_ok=True)
    with open(output_vcf, "w") as out:
        out.write(HEADER_LINES[0] + "\n")
        out.write(f"##fileDate={datetime.date.today():%Y%m%d}\n")
        out.write("##source=seqc2_sv_table_to_vcf.py\n")
        for name, length in contigs:
            out.write(f"##contig=<ID={name}" + (f",length={length}" if length else "") + ">\n")
        for line in HEADER_LINES[1:]:
            out.write(line + "\n")
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, pos, vid, ref, alt, info in records:
            out.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\tPASS\t{info}\n")

    print(f"Wrote {len(records)} VCF records from {df.height - skipped} table rows "
          f"-> {output_vcf}", file=sys.stderr)

    if compress_bgzip:
        pysam.tabix_index(str(output_vcf), preset="vcf", force=True, keep_original=True)
        print(f"bgzipped + indexed: {output_vcf}.gz", file=sys.stderr)


if __name__ == "__main__":
    inputs = [
        PROJECT_ROOT / "data" / dataset / "SEQC2_Consensus_SVs.csv",
        PROJECT_ROOT / "data" / dataset / "SEQC2_Validated_SVs.csv",
        PROJECT_ROOT / "data" / dataset / "SEQC2_PCR_SVs.csv",
    ]

    genome_fai = PROJECT_ROOT / "data/ref/Homo_sapiens_assembly38.fasta.fai"

    for input_csv in inputs:
        print("Processing: ", (input_csv.name))

        convert(
            input_csv=input_csv,
            genome_fai = genome_fai
        )
