#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import re
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bioautoml.sequence_names import sequence_id, source_info, write_names
warnings.filterwarnings("ignore")
from Bio import SeqIO


def safe_source_name(finput):
    """Return a filesystem-independent source token for generated IDs."""

    source = os.path.splitext(os.path.basename(finput))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source)


def safe_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_cleaning_record(handle, name, original, cleaned):
    removed = len(original) - len(cleaned)
    removed_symbols = ''.join(sorted(set(original) - set(cleaned)))
    handle.write(
        f"{name}\t{len(original)}\t{len(cleaned)}\t{removed}\t{removed_symbols}\n"
    )

def preprocessing_protein(finput, foutput, fset, root=None):
    # Anything not in the 20 canonical amino acids is removed
    invalid_chars = r"[^ACDEFGHIKLMNPQRSTVWY]"

    cleaning_output = f"{foutput}.cleaning.tsv"
    with open(foutput, 'w') as file, open(cleaning_output, 'w') as cleaning_file:
        cleaning_file.write(
            "nameseq\toriginal_length\tcleaned_length\tremoved_count\tremoved_symbols\n"
        )
        source = source_info(finput, fset, root)
        names = []
        for i, seq_record in enumerate(SeqIO.parse(finput, "fasta")):
            name_seq = sequence_id(finput, fset, i, seq_record.name, root, source)
            names.append(name_record(finput, fset, i, seq_record, name_seq, source))
            seq = str(seq_record.seq.upper())

            # Remove invalid amino acids and alignment hyphens
            cleaned_seq = re.sub(invalid_chars, "", seq)

            file.write(f">{name_seq}\n{cleaned_seq}\n")
            write_cleaning_record(cleaning_file, name_seq, seq, cleaned_seq)
            print(f"{name_seq}: cleaned protein sequence (non-standard amino acids removed)")
    
    write_names(finput, fset, names, root)
    print("Finished")

def preprocessing_dna(finput, foutput, fset, root=None):
    # Only A, T, G, C are valid; remove everything else
    invalid_chars = r"[^ATGCU]"

    cleaning_output = f"{foutput}.cleaning.tsv"
    with open(foutput, 'w') as file, open(cleaning_output, 'w') as cleaning_file:
        cleaning_file.write(
            "nameseq\toriginal_length\tcleaned_length\tremoved_count\tremoved_symbols\n"
        )
        source = source_info(finput, fset, root)
        names = []
        for i, seq_record in enumerate(SeqIO.parse(finput, "fasta")):
            name_seq = sequence_id(finput, fset, i, seq_record.name, root, source)
            names.append(name_record(finput, fset, i, seq_record, name_seq, source))
            seq = str(seq_record.seq.upper())

            cleaned_seq = re.sub(invalid_chars, "", seq).replace("U", "T")

            file.write(f">{name_seq}\n{cleaned_seq}\n")
            write_cleaning_record(cleaning_file, name_seq, seq, cleaned_seq)
            print(f"{name_seq}: cleaned DNA sequence (invalid nucleotides removed)")
    
    write_names(finput, fset, names, root)
    print("Finished")


def name_record(filename, set_name, index, record, internal_id, source):
    return dict(internal_id=internal_id, original_id=record.id, original_header=record.description,
                source_file=Path(filename).name, source_id=source['source_id'],
                record_number=index + 1, split=set_name.split('_', 1)[0])

#############################################################################    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Fasta format file, E.g., dataset.fasta')
    parser.add_argument('-o', '--output', help='Fasta format file, E.g., preprocessing.fasta')
    parser.add_argument('-s', '--set', default="", help='Set type; train or test')
    parser.add_argument('-d', '--data', default="", help='Data type; DNA/RNA or Protein')
    parser.add_argument('--run_root', default=os.environ.get('BIOAUTOML_RUN_ROOT'))

    args = parser.parse_args()
    finput = str(args.input)
    foutput = str(args.output)
    fset = str(args.set)
    fdata = str(args.data)

    if fdata == "DNA/RNA":
        preprocessing_dna(finput,foutput,fset,args.run_root)
    elif fdata == "Protein":
        preprocessing_protein(finput,foutput,fset,args.run_root)
#############################################################################]
