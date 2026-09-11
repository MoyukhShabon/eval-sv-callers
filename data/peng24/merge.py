# %%
import subprocess, tempfile, os, argparse, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl

# %%
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-t", "--threads", type=int, default=1,
        help="Number of parallel worker threads for the outer (per-group) loop. Default = 1"
    )
    return p.parse_args()

# %%
project_root = Path("../..").resolve()
annot_root = project_root / "annot"
data_root = project_root / "data"
dataset = "peng24"

CALLER_BLACKLIST: list[str] = [
    "svision",
]

# %%
annot = pl.read_csv(annot_root / dataset / "vcf_annotations.tsv", separator="\t")

if CALLER_BLACKLIST:
    n_before = annot.height
    annot = annot.filter(~pl.col("caller").is_in(CALLER_BLACKLIST))
    print(f"blacklist excluded {n_before - annot.height} row(s) for callers={CALLER_BLACKLIST}")

# %%
fixed_root = data_root / dataset / "vcf_fixed"
svcf_root   = data_root / dataset / "svcf"
merged_root = data_root / dataset / "merged"
svcf_root.mkdir(parents=True, exist_ok=True)
merged_root.mkdir(parents=True, exist_ok=True)

# a lock just to keep interleaved prints from different threads readable
_print_lock = threading.Lock()

def log(msg: str) -> None:
    with _print_lock:
        print(msg)

def run(cmd: list[str]) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def decompress_and_filter(in_vcf_gz: Path, caller: str, pass_only: bool = False) -> Path:
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
def process_group(platform: str, aligner: str, sample: str, annot: pl.DataFrame) -> None:
    grp = annot.filter(
        (pl.col("platform") == platform)
        & (pl.col("aligner") == aligner)
        & (pl.col("sample") == sample)
    )

    tag = f"{platform}_{sample}_{aligner}"
    log(f"\n=== group {tag} ({grp.height} callers) ===")

    svcf_paths: list[tuple[str, Path]] = []

    for row in grp.iter_rows(named=True):
        caller = row["caller"]
        in_vcf_gz = Path(fixed_root / row["filename"])
        log(f"=> [{tag}] Processing Caller: {caller} | Path: {in_vcf_gz}")

        if not in_vcf_gz.exists():
            log(f"  [skip] missing: {in_vcf_gz}")
            continue

        tmp_path = decompress_and_filter(in_vcf_gz, caller, pass_only=False)

        out_svcf = svcf_root / f"{row['filename'].split('.')[0]}.svcf"

        try:
            run([
                "octopusv", "correct",
                "-i", str(tmp_path),
                "-o", str(out_svcf),
            ])
            svcf_paths.append((caller, out_svcf))
        except subprocess.CalledProcessError as e:
            log(f"  [FAIL] correct failed for {caller}: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

    if len(svcf_paths) < 2:
        log(f"  [skip merge] [{tag}] only {len(svcf_paths)} caller(s) succeeded, need >=2")
        return

    merged_out = merged_root / f"{tag}.merged.svcf"
    run([
        "octopusv", "merge",
        "-i", *[str(p) for _, p in svcf_paths],
        "-o", str(merged_out),
        "--union", "--upsetr"
    ])
    log(f"  -> {merged_out}  callers={[c for c, _ in svcf_paths]}")


# %%
def main():
    args = parse_args()

    groups = (
        annot.select(["platform", "aligner", "sample"])
        .unique()
        .sort(["sample", "platform", "aligner"])
    )

    group_rows = list(groups.iter_rows())
    log(f"Running {len(group_rows)} group(s) with {args.threads} thread(s)")

    if args.threads <= 1:
        for platform, aligner, sample in group_rows:
            process_group(platform, aligner, sample, annot)
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futures = {
                ex.submit(process_group, platform, aligner, sample, annot): (platform, aligner, sample)
                for platform, aligner, sample in group_rows
            }
            for fut in as_completed(futures):
                platform, aligner, sample = futures[fut]
                tag = f"{platform}_{sample}_{aligner}"
                try:
                    fut.result()
                except Exception as e:
                    log(f"  [ERROR] group {tag} failed: {e}")


if __name__ == "__main__":
    main()

