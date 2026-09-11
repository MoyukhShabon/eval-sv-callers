# %%
import polars as pl
from pathlib import Path

project_root = Path("../..").resolve()

# %%
annot_root = project_root / "annot"
data_root = project_root / "data"

dataset = "peng24"

matrix_outdir = data_root / dataset / "matrix"
matrix_outdir.mkdir(exist_ok=True, parents=True)

# %%
svcf_paths = sorted(list((data_root / dataset / "merged").glob("*.merged.svcf")))

# %%
def merged_svcf_to_matrix(svcf_path):
    df = (
        pl.read_csv(
            svcf_path,
            separator="\t",
            comment_prefix="##",
            null_values=".",
            columns = ["#CHROM", "POS", "REF", "ALT", "INFO"],
            truncate_ragged_lines=True,
        )
        .rename(lambda x: x.lstrip("#").lower())
    )

    # --- pull svtype and sources out of the info string ---
    df = df.with_columns(
        pl.col("info").str.extract(r"SVTYPE=([^;]+)").alias("svtype"),
        pl.col("info").str.extract(r"SOURCES=([^;]+)").alias("sources_raw"),
    )


    # --- explode sources into one row per (variant, source_label) ---
    # source_label pattern: platform_sample_aligner_caller_sample
    # e.g. "CCS_CHM13_minimap2_cuteSV2_CHM13"
    label_pattern = r"^(?<platform>[^_]+)_(?<sample>[^_]+)_(?<aligner>[^_]+)_(?<caller>[^_]+)_(?<sample2>[^_]+)$"

    long = (
        df
        .select(["chrom", "pos", "ref", "alt", "svtype", "sources_raw"])
        .with_columns(pl.col("sources_raw").str.split(","))
        .explode("sources_raw")
        .rename({"sources_raw": "source_label"})
        .with_columns(pl.col("source_label").str.extract_groups(label_pattern).alias("parts"))
        .unnest("parts")
        .filter(pl.col("caller").is_not_null())
        # # sanity check: the two sample tokens should match; drop if not
        # .filter(pl.col("sample") == pl.col("sample2"))
        # .drop("sample2")
        .with_columns(
            pl.col("caller").str.to_lowercase(),
            pl.lit(True).alias("present"),
        )
    )


    # --- pivot to wide boolean matrix, now with platform/sample/aligner as id columns ---
    id_cols = ["chrom", "pos", "ref", "alt", "svtype", "platform", "sample", "aligner"]

    matrix = long.pivot(
        values="present",
        index=id_cols,
        on="caller",
        aggregate_function="first",
    )

    caller_cols = [c for c in matrix.columns if c not in id_cols]
    matrix = matrix.with_columns([pl.col(c).fill_null(False) for c in caller_cols])

    return matrix

# %%
for i, path in enumerate(svcf_paths):

    cluster = path.name.split(".")[0]

    print(f"{i+1}. Processing octopuSV union merge cluster: {cluster}" )

    matrix = merged_svcf_to_matrix(path)

    matrix.write_parquet(matrix_outdir / f"{cluster}.matrix.parquet",  compression="snappy")
