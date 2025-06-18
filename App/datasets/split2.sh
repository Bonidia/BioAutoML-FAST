input_file="dataset20_timmons_protein/ENNAVIA_D_dataset.fasta"
train_dir="train"
test_dir="test"

# Create directories if they don't exist
mkdir -p "$train_dir" "$test_dir"

# Create or clear output files
> "$train_dir/random-seq-non-antiviral.fasta"
> "$train_dir/anti-coronavirus.fasta"
> "$test_dir/random-seq-non-antiviral.fasta"
> "$test_dir/anti-coronavirus.fasta"

# Read the input file line by line
while IFS= read -r line; do
    if [[ "$line" == *"|random-seq-non-antiviral|"* ]]; then
        # Check if it belongs to train or test
        if [[ "$line" == *"subdataset_test" ]]; then
            # Write to test random-seq-non-antiviral file
            echo "$line" >> "$test_dir/random-seq-non-antiviral.fasta"
            # Read the next line (sequence) and write it
            IFS= read -r seq
            echo "$seq" >> "$test_dir/random-seq-non-antiviral.fasta"
        else
            # Write to train random-seq-non-antiviral file
            echo "$line" >> "$train_dir/random-seq-non-antiviral.fasta"
            # Read the next line (sequence) and write it
            IFS= read -r seq
            echo "$seq" >> "$train_dir/random-seq-non-antiviral.fasta"
        fi
    elif [[ "$line" == *"|anti-coronavirus|"* ]]; then
        # Check if it belongs to train or test
        if [[ "$line" == *"subdataset_test" ]]; then
            # Write to test anti-coronavirus file
            echo "$line" >> "$test_dir/anti-coronavirus.fasta"
            # Read the next line (sequence) and write it
            IFS= read -r seq
            echo "$seq" >> "$test_dir/anti-coronavirus.fasta"
        else
            # Write to train anti-coronavirus file
            echo "$line" >> "$train_dir/anti-coronavirus.fasta"
            # Read the next line (sequence) and write it
            IFS= read -r seq
            echo "$seq" >> "$train_dir/anti-coronavirus.fasta"
        fi
    fi
done < "$input_file"

echo "Separation complete. Files created in $train_dir and $test_dir directories."