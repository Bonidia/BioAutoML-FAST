#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import re
import argparse
warnings.filterwarnings("ignore")
from Bio import SeqIO

def preprocessing_protein(finput,foutput,fset):
    alphabet = ("B|J|O|U|X|Z")
    file = open(foutput, 'a')
    for i, seq_record in enumerate(SeqIO.parse(finput, "fasta")):
        name_seq = f"pre_{f'{i}_{fset}_' if fset else ''}" + str(seq_record.name)
        seq = seq_record.seq.upper()
        if re.search(alphabet, str(seq)) is not None:
            print(name_seq)
            print("Removed Sequence")
        else:
            file.write(f">{name_seq}")
            file.write("\n")
            file.write(str(seq).replace("-", "")) # remove hyphen from alignment
            file.write("\n")
            print(name_seq)
            print("Included Sequence")
    print("Finished")

def preprocessing_dna(finput,foutput,fset):
    alphabet = ("B|D|E|F|H|I|J|K|L|M|N|O|P|Q|R|S|V|W|X|Y|Z")
    file = open(foutput, 'a')
    for i, seq_record in enumerate(SeqIO.parse(finput, "fasta")):
        name_seq = f"pre_{f'{i}_{fset}_' if fset else ''}" + str(seq_record.name)
        seq = seq_record.seq
        if re.search(alphabet, str(seq)) is not None:
            print(name_seq)
            print("Removed Sequence")
        else:
            file.write(f">{name_seq}")
            file.write("\n")
            file.write(str(seq.back_transcribe()))
            file.write("\n")
            print(name_seq)
            print("Included Sequence")
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