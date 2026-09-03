#!/usr/bin/env python
import polars as pl
from pathlib import Path
import json

project_root = Path("../..").resolve()

## Glob VCF paths
dataset = "peng24"
vcf_paths = list((project_root / "data" / dataset / "1100VCF").glob("*.vcf.gz"))

# Formulate table
metadata = [] 

for i, path in enumerate(vcf_paths):
    tokens = path.name.removesuffix(".vcf.gz").split("_")

    record = {
        "platform": tokens[0],
        "sample": tokens[1],
        "aligner": tokens[2],
        "caller": tokens[3],
        "sample2": tokens[4],
        "filename": path.name
    }

    metadata.append(record)

metadata = pl.DataFrame(metadata)

if metadata.filter(pl.col("sample") != pl.col("sample2")).is_empty():
    metadata = metadata.drop("sample2")

metadata.write_csv("vcf_annotations.tsv", separator="\t")

# Write each unique categories to json
unique_values = {
    "platform": sorted(metadata["platform"].unique().to_list()),
    "sample":   sorted(metadata["sample"].unique().to_list()),
    "aligner":  sorted(metadata["aligner"].unique().to_list()),
    "caller":   sorted(metadata["caller"].unique().to_list()),
}

with open("categories.json", "w") as f:
    json.dump(unique_values, f, indent=4)




