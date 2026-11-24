# %%
import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("../..").resolve()

# %%
def read_vcf(vcf_path: Path) -> pl.DataFrame:
	return (
		pl.read_csv(vcf_path, separator="\t", comment_prefix="##")
		.rename(lambda x: x.strip().replace("#", "").lower()
	))

# %%
vcf: pl.DataFrame = read_vcf("S1_all_svs_validation.vcf.gz")
vcf

# %%
# Keys to extract from the INFO filed
keys: list = ["SVTYPE", "SVLEN", "END", "CHR2", "STRANDS", "CIPOS", "CIEND", "N_CALLERS", "CALLERS", "SAMPLE", "VALIDATION"]

vcf_parsed: pl.DataFrame = (
    vcf.with_columns([
        pl.col("info").str.extract(rf"(?:^|;){k}=([^;]+)").alias(k)
        for k in keys
    ])
    .with_columns(pl.col("SVLEN", "END", "N_CALLERS").cast(pl.Int64))
    .rename(lambda x: x.lower())
)

vcf_parsed

# %%
matrix: pl.DataFrame = (
    vcf_parsed
    ## Variants not validated via qPCR are does not have validation tag
    .filter(pl.col("validation").is_not_null())
    .with_columns(pl.col("callers").str.to_lowercase().str.split(","))
    .explode("callers")
    # Fill calls with True, representing SVs called by a caller
    .with_columns(pl.lit(True).alias("called"))
    # Change validation column to boolean
    .with_columns(pl.when(pl.col("validation") == "TP").then(pl.lit(True)).otherwise(False).alias("validation"))
)

# Get caller list, for their missing values in respective columns to be filled with false after pivot
caller_cols: pl.Series = matrix["callers"].unique()

# Pivot and make call matrix
matrix: pl.DataFrame = (
	matrix
    .pivot(
        on = "callers",
        values = "called",
        # Select informative columns as index
        index = ['sample', 'chrom', 'pos', 'chr2', 'end', 'ref', 'alt', 'id',  'svtype', 'svlen', 'strands', 'cipos', 'ciend', 'n_callers', "validation"]
	)
    # fill missing with False i.e not called by caller
    .with_columns(pl.col(caller_cols).fill_null(False))
)

matrix

# %%
matrix.write_parquet("jiang26.matrix.parquet", compression="snappy")


