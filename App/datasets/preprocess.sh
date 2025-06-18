#!/bin/bash

dir="dataset26_cai_dnarna/test"

# Process each fasta file in the directory
for fasta_file in "$dir"/*.fasta; do
    if [ -f "$fasta_file" ]; then
        # Get filename without path and extension
        filename=$(basename "$fasta_file" .fasta)
        
        # Create temporary file
        temp_file=$(mktemp)
        
        # Process the file to add headers
        awk -v name="$filename" '
        /^>/ {print; next}  # If line starts with >, print it (already has header)
        {print ">"name; print}  # Otherwise, add header and print sequence
        ' "$fasta_file" > "$temp_file"
        
        # Replace original file with the processed one
        mv "$temp_file" "$fasta_file"
        
        echo "Processed: $fasta_file"
    fi
done

echo "All FASTA files processed."