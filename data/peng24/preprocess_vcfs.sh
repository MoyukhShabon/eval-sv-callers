#!/usr/bin/env bash
#
# For each VCF in vcf/:
#   1. fix        -> BGZF, patch header (picky), reheader contigs + sample, sort   [temp only]
#   2. reannotate -> prefix record IDs with the caller name (e.g. "cuteSV2.<id>") -> vcf_fixed/
#   3. clean      -> octopusv clean on the vcf_fixed/ output                       -> vcf_cleaned/
#
# Expected filename format: READS_REF_ALIGNER_CALLER_REF.vcf.gz
#   e.g. CCS_CHM13_minimap2_cuteSV2_CHM13.vcf.gz -> caller = cuteSV2
set -euo pipefail
shopt -s nullglob

project_root="../.."
ref="${project_root}/data/ref/Homo_sapiens_assembly38.fasta"
ref_fai="${ref}.fai"

dataset="peng24"
outdir_fixed="${project_root}/data/${dataset}/vcf_fixed"
outdir_clean="${project_root}/data/${dataset}/vcf_cleaned"
failed_log="${project_root}/data/${dataset}/FAILED_FILES.txt"

for tool in samtools bcftools bgzip octopusv; do
	command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: $tool not found in PATH" >&2; exit 1; }
done

mkdir -p "$outdir_fixed" "$outdir_clean"
: > "$failed_log"

echo "==> Checking reference files"
[[ -f "$ref" ]] || { echo "	ERROR: Reference FASTA not found: $ref" >&2; exit 1; }
if [[ ! -f "$ref_fai" ]]; then
	echo "	.fai missing, indexing reference"
	samtools faidx "$ref"
fi
echo "	reference OK: $ref"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

# Forces reheader-safe BGZF, since some callers (e.g. cuteSV2) emit plain gzip.
to_bgzf() {
	local in="$1" out="$2"
	echo "	normalizing compression to BGZF"
	if ! zcat "$in" | bgzip > "$out"; then
		echo "	ERROR: unreadable VCF, cannot normalize: $in" >&2
		return 1
	fi
}

# Prefix each record ID with the caller name, unless the ID already starts with it.
# The match ignores case and a trailing version number, so cuteSV2 matches
# "cuteSV.BND.0" and sniffles2 matches "Sniffles2.INS.0S0".
reannotate_ids() {
	local in="$1" out="$2" caller="$3"
	bcftools view "$in" \
	| awk -F'\t' -v OFS='\t' -v caller="$caller" '
		BEGIN { key = caller; sub(/[0-9]+$/, "", key); key = tolower(key) }
		/^#/  { print; next }
		{
			if (tolower(substr($3, 1, length(key))) != key)
				$3 = ($3 == "." || $3 == "") ? caller : caller "." $3
			print
		}
	' \
	| bcftools view -Oz -o "$out"
}

process_one() {
	local vcf="$1" i="$2" total="$3"
	echo "==> [$i/$total] Processing $vcf"

	local base filename sample_name
	base=$(basename "$vcf")
	filename="${base%.vcf.gz}"
	sample_name="${filename%_*}"

	# Parse caller: everything between ALIGNER and the trailing REF
	# (rejoined with '_' in case the caller name itself contains one).
	local parts n caller=""
	IFS='_' read -ra parts <<< "$filename"
	n=${#parts[@]}
	if (( n >= 5 )) && [[ "${parts[1]}" == "${parts[n-1]}" ]]; then
		caller=$(IFS='_'; echo "${parts[*]:3:n-4}")
	fi

	## --- Step: fix (temp only) ---
	local bgzf_vcf="${tmpdir}/${filename}.bgzf.vcf.gz"

	if [[ "$caller" == "picky" ]]; then
		echo "	picky output detected -- patching header"
		local first_line
		first_line=$(zcat "$vcf" | head -n1)
		if [[ "$first_line" != "##fileformat="* ]]; then
			echo "	##fileformat line missing, injecting VCFv4.2"
			{ echo "##fileformat=VCFv4.2"; zcat "$vcf"; } \
				| bgzip -c > "${tmpdir}/${filename}.fmt.vcf.gz"
		else
			echo "	##fileformat line present, no injection needed"
			cp "$vcf" "${tmpdir}/${filename}.fmt.vcf.gz"
		fi
		to_bgzf "${tmpdir}/${filename}.fmt.vcf.gz" "$bgzf_vcf"
	else
		to_bgzf "$vcf" "$bgzf_vcf"
	fi

	# bcftools reheader -s expects a FILE containing the new sample name, not the name itself.
	local samples_file="${tmpdir}/${filename}.samples.txt"
	echo "$sample_name" > "$samples_file"

	local reheadered_vcf="${tmpdir}/${filename}.reheadered.vcf.gz"
	echo "	reheadering contigs from .fai, renaming sample to '${sample_name}'"
	bcftools reheader -f "$ref_fai" -s "$samples_file" -o "$reheadered_vcf" "$bgzf_vcf"

	# Some callers (e.g. svim) emit coordinate-unsorted records, which breaks tabix indexing.
	local sorted_vcf="${tmpdir}/${filename}.sorted.vcf.gz"
	echo "	sorting by coordinate"
	bcftools sort -Oz -o "$sorted_vcf" "$reheadered_vcf"

	## --- Step: reannotate IDs -> vcf_fixed ---
	local fixed_vcf="${outdir_fixed}/${base}"
	if [[ -n "$caller" ]]; then
		echo "	prefixing IDs with caller '${caller}'"
		reannotate_ids "$sorted_vcf" "$fixed_vcf" "$caller"
	else
		echo "	WARNING: filename doesn't match READS_REF_ALIGNER_CALLER_REF, IDs left unchanged" >&2
		mv "$sorted_vcf" "$fixed_vcf"
	fi
	bcftools index -f -t "$fixed_vcf"
	echo "	fixed VCF complete: $fixed_vcf"

	## --- Step: clean ---
	local clean_out="${outdir_clean}/${base}"
	echo "	sanitizing for Truvari/bcftools compatibility"
	octopusv clean "$fixed_vcf" "$clean_out" -g "$ref"
	echo "	clean VCF complete: $clean_out"

	echo "==> [$i/$total] Finished ${filename}"
}

vcfs=(vcf/*.vcf.gz)
total=${#vcfs[@]}
(( total > 0 )) || { echo "ERROR: no *.vcf.gz files found in vcf/" >&2; exit 1; }

i=0
for vcf in "${vcfs[@]}"; do
	i=$((i + 1))
	if ! ( process_one "$vcf" "$i" "$total" ); then
		echo "	!! FAILED: $vcf -- skipping (see error above)" >&2
		echo "$vcf" >> "$failed_log"
	fi
done

echo "==> All done. Processed ${total} VCF files."
if [[ -s "$failed_log" ]]; then
	echo "==> Some files failed, see: $failed_log"
fi