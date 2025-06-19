#!/bin/bash

# Check if input file is provided

input_file="Dataset_final.txt"
p_file="hot.fasta"
n_file="cold.fasta"

# Create or clear output files
> "$p_file"
> "$n_file"

# Read the input file line by line
while IFS= read -r line; do
    if [[ "$line" == *"Hot"* ]]; then
        # Write to P file
        echo "$line" >> "$p_file"
        # Read the next line (sequence) and write it
        IFS= read -r seq
        echo "$seq" >> "$p_file"
    elif [[ "$line" == *"Cold"* ]]; then
        # Write to N file
        echo "$line" >> "$n_file"
        # Read the next line (sequence) and write it
        IFS= read -r seq
        echo "$seq" >> "$n_file"
    fi
done < "$input_file"

echo "Separation complete. P sequences saved to $p_file, N sequences saved to $n_file"