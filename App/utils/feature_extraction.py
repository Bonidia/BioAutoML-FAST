from bioautoml.sequence_names import register_sources, source_info
from bioautoml.execution import run_root
from bioautoml.execution import run_path, timed, start_cli
import os
import shutil
import subprocess
import sys

import pandas as pd


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# The web process starts in App; the CLI starts in the project directory.
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)
from bioautoml.feature_execution import run_feature_commands as run_descriptor_commands
SELF_PREFIXED_DATASETS = {"iFeature-features", "repDNA"}


def feature_dataset_name(dataset):
    return os.path.splitext(os.path.basename(dataset))[0]


def read_feature_dataset(dataset):
    """Read one extractor output and apply schema-v2 feature names."""

    descriptor = feature_dataset_name(dataset)
    dataframe = pd.read_csv(dataset)
    dataframe = dataframe[dataframe["nameseq"].astype(str) != "nameseq"]
    if dataframe["nameseq"].duplicated().any():
        raise ValueError(f"Duplicate sequence identifiers in {dataset}.")
    if descriptor not in SELF_PREFIXED_DATASETS:
        dataframe.rename(
            columns={
                column: f"{descriptor}__{column}"
                for column in dataframe.columns
                if column not in {"nameseq", "label"}
            },
            inplace=True,
        )
    return dataframe


def combine_feature_datasets(datasets):
    """Join descriptor outputs by sequence ID and preserve the first-file order."""

    dataframes = [read_feature_dataset(dataset) for dataset in datasets]
    metadata = dataframes[0].loc[:, ["nameseq", "label"]].set_index("nameseq")
    feature_frames = []

    for dataset, dataframe in zip(datasets, dataframes):
        indexed = dataframe.set_index("nameseq")
        missing = metadata.index.difference(indexed.index)
        extra = indexed.index.difference(metadata.index)
        if len(missing) or len(extra):
            raise ValueError(
                f"Sequence IDs do not align in {dataset}; "
                f"missing={list(missing[:5])}, extra={list(extra[:5])}."
            )
        indexed = indexed.reindex(metadata.index)
        if not indexed["label"].astype(str).equals(metadata["label"].astype(str)):
            raise ValueError(f"Sequence labels do not align in {dataset}.")
        feature_frames.append(indexed.drop(columns="label"))

    combined = pd.concat([metadata, *feature_frames], axis=1).reset_index()
    if combined.columns.duplicated().any():
        duplicates = combined.columns[combined.columns.duplicated()].tolist()
        raise ValueError(f"Feature schema contains duplicate columns: {duplicates[:10]}")
    return combined


def get_selected_descriptors(model):
    """Return the descriptor groups selected when the model was trained."""

    descriptor_values = model["descriptors"].iloc[0]
    return [
        descriptor
        for descriptor, selected in descriptor_values.items()
        if int(selected) == 1
    ]


def run_feature_commands(commands, log_path, n_cpu=-1):
    """Run independent descriptor commands concurrently and check their status."""

    run_descriptor_commands(commands, log_path, n_cpu, cwd=PROJECT_PATH)


@timed('test_preprocessing')
def prepare_test_fastas(test_data, data_type, path):
    """Preprocess uploaded test FASTA files in their existing label order."""

    register_sources(run_root(path), list(test_data.values()), list(test_data), 'test')
    preprocessed_fastas = []
    for label, fasta_file in test_data.items():
        preprocessed_fasta = os.path.join(path, source_info(fasta_file, f"test_{label}", run_root(path))["output_name"])
        subprocess.run(
            [
                "python",
                "other-methods/preprocessing.py",
                "-d",
                data_type,
                "-i",
                fasta_file,
                "-o",
                preprocessed_fasta,
                "--run_root", run_root(path),
                "-s",
                f"test_{label}",
            ],
            cwd=PROJECT_PATH,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )
        preprocessed_fastas.append((preprocessed_fasta, label))

    return preprocessed_fastas


def save_selected_test(datasets, model, feat_path, job_path):
    """Combine extracted groups and save columns in the model's training order."""

    dataframes = combine_feature_datasets(datasets)

    train_columns = list(model['column_train'] if 'column_train' in model else model['train'].columns)
    missing_columns = [
        column for column in train_columns if column not in dataframes.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing selected test features: {missing_columns[:10]}")

    dataframes[["nameseq"]].to_csv(
        os.path.join(feat_path, "fnameseqtest.csv"), index=False
    )
    dataframes[["label"]].to_csv(
        os.path.join(feat_path, "flabeltest.csv"), index=False
    )

    path_bio = run_path(job_path, "best_descriptors")
    os.makedirs(path_bio, exist_ok=True)
    dataframes.loc[:, train_columns].to_csv(
        os.path.join(path_bio, "best_test.csv"), index=False
    )


def extract_aminoacid_features(preprocessed_fastas, selected, selected_path, n_cpu=-1):
    """Extract only selected amino-acid descriptor groups."""

    datasets = []
    ifeature_names = [
        "CKSAAP",
        "DDE",
        "GAAC",
        "CKSAAGP",
        "GDPC",
        "GTPC",
        "CTDC",
        "CTDT",
        "CTDD",
        "CTriad",
    ]
    selected_ifeature = [
        descriptor for descriptor in ifeature_names if descriptor in selected
    ]

    for preprocessed_fasta, label in preprocessed_fastas:
        command_map = {
            "Shannon": [
                "python", "other-methods/EntropyClass.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "Shannon.csv"), "-l", label,
                "-k", "5", "-e", "Shannon",
            ],
            "Tsallis_23": [
                "python", "other-methods/TsallisEntropy.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "Tsallis_23.csv"), "-l", label,
                "-k", "5", "-q", "2.3",
            ],
            "Tsallis_30": [
                "python", "other-methods/TsallisEntropy.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "Tsallis_30.csv"), "-l", label,
                "-k", "5", "-q", "3.0",
            ],
            "Tsallis_40": [
                "python", "other-methods/TsallisEntropy.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "Tsallis_40.csv"), "-l", label,
                "-k", "5", "-q", "4.0",
            ],
            "ComplexNetworks": [
                "python", "MathFeature/methods/ComplexNetworksClass-v2.py", "-i",
                preprocessed_fasta, "-o", os.path.join(selected_path, "ComplexNetworks.csv"),
                "-l", label, "-k", "3",
            ],
            "kGap": [
                "python", "MathFeature/methods/Kgap.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "kGap.csv"), "-l", label,
                "-k", "1", "-bef", "1", "-aft", "1", "-seq", "3",
            ],
            "AAC": [
                "python", "other-methods/ExtractionTechniques-Protein.py", "-i",
                preprocessed_fasta, "-o", os.path.join(selected_path, "AAC.csv"),
                "-l", label, "-t", "AAC",
            ],
            "DPC": [
                "python", "other-methods/ExtractionTechniques-Protein.py", "-i",
                preprocessed_fasta, "-o", os.path.join(selected_path, "DPC.csv"),
                "-l", label, "-t", "DPC",
            ],
            "Global": [
                "python", "other-methods/modlAMP-modified/descriptors.py", "-option",
                "global", "-label", label, "-input", preprocessed_fasta,
                "-output", os.path.join(selected_path, "Global.csv"),
            ],
            "Peptide": [
                "python", "other-methods/modlAMP-modified/descriptors.py", "-option",
                "peptide", "-label", label, "-input", preprocessed_fasta,
                "-output", os.path.join(selected_path, "Peptide.csv"),
            ],
        }

        commands = []
        for descriptor, command in command_map.items():
            if descriptor in selected:
                commands.append((descriptor, command))
                dataset = os.path.join(selected_path, f"{descriptor}.csv")
                if dataset not in datasets:
                    datasets.append(dataset)

        if selected_ifeature:
            dataset = os.path.join(selected_path, "iFeature-features.csv")
            commands.append(
                (
                    "iFeature-features",
                    [
                        "python", "other-methods/iFeature-modified/iFeature.py",
                        "--file", preprocessed_fasta, "--type", *selected_ifeature,
                        "--label", label, "--out", dataset,
                    ],
                )
            )
            if dataset not in datasets:
                datasets.append(dataset)

        run_feature_commands(commands, os.path.join(selected_path, "logs"), n_cpu)

    text_input = "\n".join(
        f"{fasta_file}\n{label}" for fasta_file, label in preprocessed_fastas
    )
    for descriptor, file_name, representation in [
        ("Fourier_Integer", "Fourier_Integer.csv", "6"),
        ("Fourier_EIIP", "Fourier_EIIP.csv", "8"),
    ]:
        if descriptor not in selected:
            continue

        dataset = os.path.join(selected_path, file_name)
        subprocess.run(
            [
                "python", "other-methods/mathfeature-modified/methods/Mappings-Protein.py", "-n",
                str(len(preprocessed_fastas)), "-o", dataset, "-r", representation,
            ],
            cwd=PROJECT_PATH,
            text=True,
            input=text_input,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )
        datasets.append(dataset)

    return datasets


def extract_nucleotide_features(preprocessed_fastas, selected, selected_path, n_cpu=-1):
    """Extract only selected nucleotide descriptor groups."""

    datasets = []
    repdna_names = [
        "Revkmer", "PseDNC", "PseKNC", "SC-PseDNC", "SC-PseTNC",
        "DAC", "TAC", "TCC", "TACC",
    ]
    selected_repdna = [
        descriptor for descriptor in repdna_names if descriptor in selected
    ]

    for preprocessed_fasta, label in preprocessed_fastas:
        command_map = {
            "NAC": [
                "python", "MathFeature/methods/ExtractionTechniques.py", "-i",
                preprocessed_fasta, "-o", os.path.join(selected_path, "NAC.csv"),
                "-l", label, "-t", "NAC", "-seq", "1",
            ],
            "DNC": [
                "python", "MathFeature/methods/ExtractionTechniques.py", "-i",
                preprocessed_fasta, "-o", os.path.join(selected_path, "DNC.csv"),
                "-l", label, "-t", "DNC", "-seq", "1",
            ],
            "TNC": [
                "python", "MathFeature/methods/ExtractionTechniques.py", "-i",
                preprocessed_fasta, "-o", os.path.join(selected_path, "TNC.csv"),
                "-l", label, "-t", "TNC", "-seq", "1",
            ],
            "kGap_di": [
                "python", "MathFeature/methods/Kgap.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "kGap_di.csv"), "-l", label,
                "-k", "1", "-bef", "1", "-aft", "2", "-seq", "1",
            ],
            "kGap_tri": [
                "python", "MathFeature/methods/Kgap.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "kGap_tri.csv"), "-l", label,
                "-k", "1", "-bef", "1", "-aft", "3", "-seq", "1",
            ],
            "ORF": [
                "python", "MathFeature/methods/CodingClass.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "ORF.csv"), "-l", label,
            ],
            "Fickett": [
                "python", "other-methods/mathfeature-modified/methods/FickettScore.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "Fickett.csv"), "-l", label,
                "-seq", "1",
            ],
            "Shannon": [
                "python", "other-methods/EntropyClass.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "Shannon.csv"), "-l", label,
                "-k", "5", "-e", "Shannon",
            ],
            "FourierBinary": [
                "python", "other-methods/mathfeature-modified/methods/FourierClass.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "FourierBinary.csv"), "-l", label,
                "-r", "1",
            ],
            "FourierComplex": [
                "python", "other-methods/FourierClass.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "FourierComplex.csv"), "-l", label,
                "-r", "6",
            ],
            "Tsallis": [
                "python", "other-methods/TsallisEntropy.py", "-i", preprocessed_fasta,
                "-o", os.path.join(selected_path, "Tsallis.csv"), "-l", label,
                "-k", "5", "-q", "2.3",
            ],
        }

        commands = []
        for descriptor, command in command_map.items():
            if descriptor in selected:
                commands.append((descriptor, command))
                dataset = os.path.join(selected_path, f"{descriptor}.csv")
                if dataset not in datasets:
                    datasets.append(dataset)

        if selected_repdna:
            dataset = os.path.join(selected_path, "repDNA.csv")
            commands.append(
                (
                    "repDNA",
                    [
                        "python", "other-methods/repDNA/repDNA-feat.py", "--file",
                        preprocessed_fasta, "--output", dataset, "--label", label,
                        "--descriptors", *selected_repdna,
                    ],
                )
            )
            if dataset not in datasets:
                datasets.append(dataset)

        run_feature_commands(commands, os.path.join(selected_path, "logs"), n_cpu)

    return datasets


@timed('test_features')
def test_extraction(job_path, test_data, model, data_type, n_cpu=-1):
    """Extract only the descriptor groups required by a saved sequence model."""

    feat_path = run_path(job_path, "feat_extraction")
    path = os.path.join(feat_path, "test")
    selected_path = os.path.join(feat_path, "selected_test_features")
    shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(selected_path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    os.makedirs(selected_path, exist_ok=True)

    selected = get_selected_descriptors(model)
    preprocessed_fastas = prepare_test_fastas(test_data, data_type, path)

    if data_type == "DNA/RNA":
        datasets = extract_nucleotide_features(
            preprocessed_fastas, selected, selected_path, n_cpu
        )
    elif data_type == "Protein":
        datasets = extract_aminoacid_features(
            preprocessed_fastas, selected, selected_path, n_cpu
        )
    else:
        raise ValueError(f"Unsupported sequence data type: {data_type}")

    if not datasets:
        raise ValueError("The model does not contain any selected descriptor groups.")

    save_selected_test(datasets, model, feat_path, job_path)
