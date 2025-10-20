#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import re
import argparse
warnings.filterwarnings("ignore")
from Bio import SeqIO

def preprocessing_protein(finput, foutput, fset):
    # Anything not in the 20 canonical amino acids is removed
    invalid_chars = r"[^ACDEFGHIKLMNPQRSTVWY]"
    
    with open(foutput, 'a') as file:
        for i, seq_record in enumerate(SeqIO.parse(finput, "fasta")):
            name_seq = f"pre_{f'{i}_{fset}_' if fset else f'{i}_'}{seq_record.name}"
            seq = str(seq_record.seq.upper())

            # Remove invalid amino acids and alignment hyphens
            cleaned_seq = re.sub(invalid_chars, "", seq)

            file.write(f">{name_seq}\n{cleaned_seq}\n")
            print(f"{name_seq}: cleaned protein sequence (non-standard amino acids removed)")
    
    print("Finished")

def preprocessing_dna(finput, foutput, fset):
    # Only A, T, G, C are valid; remove everything else
    invalid_chars = r"[^ATGCU]"
    
    with open(foutput, 'a') as file:
        for i, seq_record in enumerate(SeqIO.parse(finput, "fasta")):
            name_seq = f"pre_{f'{i}_{fset}_' if fset else f'{i}_'}{seq_record.name}"
            seq = str(seq_record.seq.upper())

            cleaned_seq = re.sub(invalid_chars, "", seq)
            cleaned_seq = cleaned_seq.replace("U", "T")

            file.write(f">{name_seq}\n{cleaned_seq}\n")
            print(f"{name_seq}: cleaned DNA sequence (invalid nucleotides removed)")
    
    print("Finished")

#############################################################################    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Fasta format file, E.g., dataset.fasta')
    parser.add_argument('-o', '--output', help='Fasta format file, E.g., preprocessing.fasta')
    parser.add_argument('-s', '--set', default="", help='Set type; train or test')
    parser.add_argument('-d', '--data', default="", help='Data type; DNA/RNA or Protein')

    args = parser.parse_args()
    finput = str(args.input)
    foutput = str(args.output)
    fset = str(args.set)
    fdata = str(args.data)

    if fdata == "DNA/RNA":
        preprocessing_dna(finput,foutput,fset)
    elif fdata == "Protein":
        preprocessing_protein(finput,foutput,fset)
#############################################################################]