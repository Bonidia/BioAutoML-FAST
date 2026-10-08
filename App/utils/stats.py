from bioautoml.execution import run_path, timed, start_cli
import streamlit as st
import os, shutil
import re
import glob
import json
from pathlib import Path
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
import polars as pl
import pandas as pd
import numpy as np

def summary_stats(path_folder, data_type, job_path, structured):

    if structured:
        structured_file = os.path.join(path_folder, os.listdir(path_folder)[0])

        df_struct = pl.read_csv(structured_file)

        if "label" in df_struct.columns:
            df_count = df_struct["label"].value_counts()
            df_count = df_count.rename({"label": "class", "count": "num_samples"})
        else:
            struct_stats = {"class": [], "num_samples": []}

            struct_stats["class"].append("Predicted")
            struct_stats["num_samples"].append(len(df_struct))

            df_count = pl.DataFrame(struct_stats)

        df_count.write_csv(run_path(job_path, "train_stats.csv" if "train" in path_folder else "test_stats.csv"))
    else:
        source_path = Path(job_path) / 'reports/sequence_sources.json'
        sources = json.loads(source_path.read_text()) if source_path.exists() else {}
        split = Path(path_folder).name
        source_labels = {source['output_name']: source['label'] for source in sources.values()
                         if 'label' in source and (split not in ('train', 'test') or source.get('split') == split)}
        fasta_files = {}
        for filename in sorted(os.listdir(path_folder)):
            if filename.lower().endswith(('.fa', '.fasta', '.fna', '.faa')):
                label = source_labels.get(filename, Path(filename).stem.removeprefix('pre_'))
                fasta_files.setdefault(label, []).append(os.path.join(path_folder, filename))
        
        seq_stats = {"class": [], "num_seqs": [], "min_length": [], "max_length": [], 
                    "avg_length": [], "std_length": [], "sum_length": [], 
                    "Q1": [], "Q2": [], "Q3": [], "N50": []}

        if data_type == "DNA/RNA":
            seq_stats["gc_content"] = []

        for seq_class in fasta_files:
            seq_stats["class"].append(seq_class)
            
            lengths, gcs = [], []
            for filename in fasta_files[seq_class]:
                for record in SeqIO.parse(filename, "fasta"):
                    lengths.append(len(record.seq))
                    gcs.append(gc_fraction(record.seq))
            
            num_seqs = len(lengths)

            seq_stats["num_seqs"].append(num_seqs)
            seq_stats["min_length"].append(min(lengths))
            seq_stats["max_length"].append(max(lengths))
            seq_stats["avg_length"].append(np.mean(lengths))
            seq_stats["std_length"].append(np.std(lengths))
            seq_stats["sum_length"].append(sum(lengths))
            seq_stats["Q1"].append(np.percentile(lengths, 25))
            seq_stats["Q2"].append(np.percentile(lengths, 50))
            seq_stats["Q3"].append(np.percentile(lengths, 75))

            lengths.sort(reverse=True)
            total_length = sum(lengths)
            cumulative_length = 0
            for length in lengths:
                cumulative_length += length
                if cumulative_length >= total_length / 2:
                    seq_stats["N50"].append(length)
                    break
            
            if data_type == "DNA/RNA":
                seq_stats["gc_content"].append((sum(gcs) / num_seqs) * 100)
        
        pl.DataFrame(seq_stats).write_csv(run_path(job_path, "train_stats.csv" if "train" in path_folder else "test_stats.csv"))
