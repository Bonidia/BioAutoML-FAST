import streamlit as st
import os, shutil
import re
import glob
from Bio import SeqIO

def process_files(train_files, test_files, job_path, seq_type):
    alphabets = {'nt': re.compile('^[acgtu]*$', re.I), 
                'aa': re.compile('^[acdefghiklmnpqrstvwy]*$', re.I),
                'aa_exclusive': re.compile('[defhiklmpqrsvwy]', re.I)}

    train_path = os.path.join(job_path, 'train')
    os.makedirs(train_path)

    for file in train_files:
        save_path = os.path.join(train_path, file.name)
        with open(save_path, mode='wb') as f:
            f.write(file.getvalue())

    train_fasta = {os.path.splitext(f)[0] : os.path.join(train_path, f) for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))}
    
    for seq_class in train_fasta:
        pre_file = os.path.join(train_path, f"processed_{seq_class}.fasta")

        with open(pre_file, mode='a') as f:
            for record in SeqIO.parse(train_fasta[seq_class], 'fasta'):

                if seq_type == "DNA/RNA":
                    if alphabets['aa_exclusive'].search(str(record.seq)) is not None:
                        st.error("Inconsistent sequence file.")
                        return [], ""
                    if alphabets['nt'].search(str(record.seq)) is not None:
                        f.write(f">{record.id}\n")
                        f.write(f"{record.seq}\n")
                else: 
                    if alphabets['aa_exclusive'].search(str(record.seq)) is None:
                        st.error("Inconsistent sequence file.")
                        return [], ""
                    if alphabets['aa'].search(str(record.seq)) is not None:
                        f.write(f">{record.id}\n")
                        f.write(f"{record.seq}\n")

        train_fasta[seq_class] = pre_file

    test_fasta = None

    if test_files:
        test_path = os.path.join(job_path, 'test')
        os.makedirs(test_path)

        for file in test_files:
            save_path = os.path.join(test_path, file.name)
            with open(save_path, mode='wb') as f:
                f.write(file.getvalue())

        test_fasta = {os.path.splitext(f)[0] : os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))}
        
        for seq_class in test_fasta:
            pre_file = os.path.join(test_path, f"processed_{seq_class}.fasta")

            with open(pre_file, mode='a') as f:
                for record in SeqIO.parse(test_fasta[seq_class], 'fasta'):

                    if seq_type == "DNA/RNA":
                        if alphabets['aa_exclusive'].search(str(record.seq)) is not None:
                            st.error("Inconsistent sequence file.")
                            return
                        if alphabets['nt'].search(str(record.seq)) is not None:
                            f.write(f">{record.id}\n")
                            f.write(f"{record.seq}\n")
                    else: 
                        if alphabets['aa_exclusive'].search(str(record.seq)) is None:
                            st.error("Inconsistent sequence file.")
                            return
                        if alphabets['aa'].search(str(record.seq)) is not None:
                            f.write(f">{record.id}\n")
                            f.write(f"{record.seq}\n")

            test_fasta[seq_class] = pre_file

    return train_fasta, test_fasta, seq_type