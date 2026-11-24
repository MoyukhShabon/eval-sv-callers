# %%
import polars as pl
from pathlib import Path

# %%
table = pl.read_excel("mmc3.xlsx", infer_schema_length=1000)
table.write_csv("colo829.tsv", separator="\t")

# %%
matrix = (
    table
    .rename({"tenx": "10x"})
    .select(['svid', 'chr1', 'pos1', 'chr2', 'pos2', 'svtype', 'decision', 'ori', 'svlen', 'original_set', 'illumina', 'pacbio', '10x', 'nanopore', 'pcr', 'capture', 'bionano', 'decision_somatic'])
    .filter((pl.col("decision") != "Duplicated"))
    .with_columns(
        pl.col(['illumina', 'pacbio', '10x', 'nanopore', 'pcr', 'capture', 'bionano', 'decision_somatic']).cast(pl.Boolean),
        pl.when(pl.col("decision") == "Somatic")
			.then(True)
			.otherwise(False)
			.alias("curated_somatic_truth"),
        pl.col("svtype").replace({"INV3": "INV", "INV5": "INV"})
    )
)

matrix

# %%
matrix_somatic = matrix.filter(pl.col("decision_somatic"))
matrix_somatic

# %%
matrix.write_parquet("colo829_curated-svs_germline+somatic.parquet", compression="snappy")
matrix_somatic.write_parquet("colo829_curated-svs_somatic-only.parquet", compression="snappy")



