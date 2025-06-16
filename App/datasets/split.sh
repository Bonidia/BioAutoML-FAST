#!/bin/bash

# Check if input file is provided


input_file="dataset11_lv_dnarna/train/S.cerevisiae.fasta"
p_file="positive.fasta"
n_file="negative.fasta"

# Create or clear output files
> "$p_file"
> "$n_file"

# Read the input file line by line
while IFS= read -r line; do
    if [[ "$line" == ">+"* ]]; then
        # Write to P file
        echo "$line" >> "$p_file"
        # Read the next line (sequence) and write it
        IFS= read -r seq
        echo "$seq" >> "$p_file"
    elif [[ "$line" == ">-"* ]]; then
        # Write to N file
        echo "$line" >> "$n_file"
        # Read the next line (sequence) and write it
        IFS= read -r seq
        echo "$seq" >> "$n_file"
    fi
done < "$input_file"

echo "Separation complete. P sequences saved to $p_file, N sequences saved to $n_file"