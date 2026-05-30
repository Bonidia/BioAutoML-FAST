![Python](https://img.shields.io/badge/python-v3.11-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Status](https://img.shields.io/badge/status-up-brightgreen)

<h1 align="center">
  <img src="https://raw.githubusercontent.com/Bonidia/BioAutoML-FAST/refs/heads/main/App/imgs/logo.png" alt="BioAutoML-FAST" width="400">
</h1>

<h4 align="center">BioAutoML-FAST: Empowering Breakthroughs in Life Sciences with End-to-End Machine Learning</h4>

<p align="center">
  <a href="https://github.com/Bonidia/BioAutoML-FAST/">Home</a> •
  <a href="https://bioautoml.icmc.usp.br/">Web Platform</a> •
  <a href="#installing-dependencies-and-package">Installing</a> •
  <a href="#web-application">Web Application</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#trained-models">Trained Models</a> •
  <a href="#citation">Citation</a>
</p>

## Awards

⭐ 2025 Google PhD Fellowship in Health Research awarded to support outstanding and innovative research in computer science and related fields, providing total funding of USD 30.000 over two years — [[Link](https://research.google/programs-and-events/phd-fellowship/recipients/?filtertab=2025)]

⭐ ISME Scholar Mobility Fund awarded with funding of € 2.300 for a research period in July 2026 at the Helmholtz Centre for Environmental Research (UFZ) in Leipzig, Germany

## Abstract

The prediction of biological sequence properties has traditionally relied on alignment-based methods that assume evolutionary homology and depend on curated reference databases. This, in turn, limits scalability and sensitivity for large or heterogeneous datasets, remote homologs, short sequences, and rapidly evolving genomic regions. Although Machine-Learning (ML) approaches offer alignment-free alternatives, their broader adoption is limited by: (i) the lack of standardized, externally validated benchmark models across diverse datasets, and (ii) the technical expertise required for feature engineering, model selection, and evaluation. Automated machine learning (AutoML) alleviates these challenges by systematically optimizing representations and models with minimal user intervention. However, most existing frameworks prioritize task-specific model construction and lack mechanisms for preserving trained models as persistent, comparable benchmarks. We introduce BioAutoML-FAST, an end-to-end web platform for automated ML analysis of nucleotide and amino acid sequences. It supports both classification and regression tasks and automates feature extraction, model training, and evaluation without requiring prior user expertise. Uniquely, it serves as a community benchmarking resource, hosting a continuously expanding repository of reusable, standardized models (currently 60) for genomic, transcriptomic, and proteomic applications. Extensive validation on independent datasets demonstrates performance comparable to or exceeding that of state-of-the-art methods, including protein language models such as ESM-2. BioAutoML-FAST is available at https://bioautoml.icmc.usp.br/. This website is free and open to all users, and there is no login requirement.

<h1 align="center">
  <img src="https://raw.githubusercontent.com/Bonidia/BioAutoML-FAST/refs/heads/main/App/imgs/overview.png" alt="Overview" width="600">
</h1>

## Key Features

- **Alignment-free** machine learning for nucleotide and amino acid sequences
- **Automated feature engineering** — Bayesian optimization (Optuna) selects the best descriptor combination from a pool of 20 nucleotide descriptors and 23 protein descriptors
- **Automated model training and hyperparameter optimization** — supports LightGBM, XGBoost, and Random Forest with Optuna-driven tuning
- **Classification and regression** — binary, multiclass, and quantitative prediction tasks
- **Structured data support** — `generation.py` can be used directly with pre-computed feature matrices (CSV), without FASTA input
- **Pre-trained model repository** — 60+ community benchmarks spanning genomics, transcriptomics, and proteomics, browsable on the web platform
- **Reusable models** — trained models can be saved and re-applied to new sequences for prediction
- **Web platform** — hosted at [bioautoml.icmc.usp.br](https://bioautoml.icmc.usp.br/), no login required; can also be self-hosted

## Authors

* Breno L. S. de Almeida, Robson P. Bonidia, Martin Bole, Anderson P. Avila-Santos, Peter F. Stadler, Ulisses Rocha, André C. P. L. F. de Carvalho

* **Correspondence:** brenoslivio@usp.br, bonidia@utfpr.edu.br or ulisses.rocha@ufz.de

## Publication

Silva de Almeida, B. L., Bonidia, R., Bole, M., Avila-Santos, A., Stadler, P. F., Nunes da Rocha, U., & de Carvalho, A. C. L. F. (2026). BioAutoML-FAST: an automated machine-learning platform for reusable and benchmarked biological sequence models. bioRxiv, 2026-04. [DOI](https://doi.org/10.64898/2026.04.18.719383)

## Installing dependencies and package

If you want to use BioAutoML-FAST locally you can clone the repository and add the necessary submodules:

```sh
git clone https://github.com/Bonidia/BioAutoML-FAST.git BioAutoML-FAST

cd BioAutoML-FAST

git submodule init

git submodule update
```

### uv (Linux/Mac/Windows)

**1 - Install uv** 

If using Linux or Mac:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If using Windows, use irm to download the script and execute it with iex:

```sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2 - Preparing the virtual environment**

With uv installed, inside the folder use following command to synchronize the virtual environment with the necessary dependencies:

```sh
uv sync
```

**3 - Activate environment**

After preparing the environment, you can activate the environment on Linux or Mac with:

```sh
source .venv/bin/activate
```

Using Windows:

```sh
.venv\Scripts\activate
```

**4 - Deactivate environment**

You can deactivate the environment using:

```sh
deactivate
```

## Web Application

The hosted platform is freely available at **https://bioautoml.icmc.usp.br/** — no login required.

If you prefer to run the web application locally or deploy it on your own server, follow the steps below.

### Requirements

The web app uses [Streamlit](https://streamlit.io/) for the interface and [Redis](https://redis.io/) + [RQ](https://python-rq.org/) to handle background jobs. Make sure Redis is installed and running before starting the app:

```sh
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server

# macOS (Homebrew)
brew install redis
brew services start redis
```

### Running the web app

With the virtual environment activated and Redis running, open **two separate terminals** from the repository root:

**Terminal 1 — start the RQ worker:**

```sh
cd App
rq worker bioautoml
```

**Terminal 2 — start the Streamlit server:**

```sh
cd App
streamlit run app.py
```

The app will be available at `http://localhost:8501` by default.

### Deploying as a system service (Linux)

For a persistent server deployment, you can use the provided systemd service files located in `App/services/`. Copy them to your systemd directory and enable them:

```sh
sudo cp App/services/bioautoml-web.service /etc/systemd/system/
sudo cp App/services/bioautoml-worker.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable bioautoml-web bioautoml-worker
sudo systemctl start bioautoml-web bioautoml-worker
```

## How to use

BioAutoML-FAST uses a two-step pipeline: `engineering.py` handles feature extraction and descriptor selection, then automatically invokes `generation.py` for model training and hyperparameter optimization.

<h1 align="center">
  <img src="https://raw.githubusercontent.com/Bonidia/BioAutoML-FAST/refs/heads/main/App/imgs/modules.png" alt="Modules" width="600">
</h1>

### `engineering.py` 

The `engineering.py` script performs the first step of BioAutoML-FAST. It extracts sequence descriptors from the input FASTA files, performs automated feature engineering/descriptor selection, and then automatically calls `generation.py` for model generation and hyperparameter optimization.

| Option | Description | Default |
|---|---|---|
| `-fasta_train`, `--fasta_train` | One or more training FASTA files. | Required |
| `-fasta_label_train`, `--fasta_label_train` | Labels associated with each training FASTA file. The order must match `-fasta_train`. | Required |
| `-fasta_test`, `--fasta_test` | One or more testing FASTA files. | Optional |
| `-fasta_label_test`, `--fasta_label_test` | Labels associated with each testing FASTA file. The order must match `-fasta_test`. | Optional |
| `-dtype`, `--dtype` | Type of input data. Supported values: `DNA/RNA` or `Protein`. | `DNA/RNA` |
| `-task`, `--task` | Machine learning task. Use `0` for classification and `1` for regression. | `0` |
| `-estimations`, `--estimations` | Number of estimations used during automated feature engineering. | `200` |
| `-patience`, `--patience` | Number of trials without improvement before early stopping. | `80` |
| `-tuning`, `--tuning` | Number of trials used for hyperparameter optimization in `generation.py`. | `150` |
| `-difference`, `--difference` | Minimum improvement required before early stopping. | `0.001` |
| `-n_cpu`, `--n_cpu` | Number of CPU cores to use. Use `-1` to use all available cores. | `-1` |
| `-output`, `--output` | Output directory where results will be saved. | Required |

#### Example: DNA/RNA (nucleotide) classification

```sh
python engineering.py \
  -fasta_train train/ncRNA.fasta train/lncRNA.fasta train/circRNA.fasta \
  -fasta_label_train ncRNA lncRNA circRNA \
  -fasta_test test/ncRNA.fasta test/lncRNA.fasta test/circRNA.fasta \
  -fasta_label_test ncRNA lncRNA circRNA \
  -dtype DNA/RNA \
  -task 0 \
  -output results
```

#### Example: Protein (amino acid) regression

```sh
python engineering.py \
  -fasta_train train/enzyme.fasta \
  -fasta_label_train enzyme \
  -fasta_test test/enzyme.fasta \
  -fasta_label_test enzyme \
  -dtype Protein \
  -task 1 \
  -output results
```

### `generation.py`

The `generation.py` script performs the second step of BioAutoML-FAST. It trains and optimizes machine learning models using the descriptors generated during the feature engineering step. The module supports both classification and regression tasks, including hyperparameter optimization and external test evaluation.

> **Structured data:** `generation.py` can also be used as a standalone script with any pre-computed feature matrix in CSV format — no FASTA input or feature extraction required. This makes it suitable for general tabular ML tasks beyond biological sequences.

| Option | Description | Default |
|---|---|---|
| `-path_model`, `--path_model` | Path to a previously trained model to be reused for prediction or evaluation. | `''` |
| `-task`, `--task` | Machine learning task. Use `0` for classification and `1` for regression. | `0` |
| `-tuning`, `--tuning` | Number of hyperparameter optimization trials. | `150` |
| `-train`, `--train` | Training feature matrix in CSV format. | Required |
| `-train_label`, `--train_label` | Training labels in CSV format. | Required |
| `-train_nameseq`, `--train_nameseq` | CSV file containing sequence names/identifiers for the training set. | Required |
| `-test`, `--test` | Test feature matrix in CSV format. | Optional |
| `-test_label`, `--test_label` | Test labels in CSV format. | Optional |
| `-test_nameseq`, `--test_nameseq` | CSV file containing sequence names/identifiers for the test set. | Optional |
| `-n_cpu`, `--n_cpu` | Number of CPU cores to use. Use `-1` to use all available cores. | `-1` |
| `-output`, `--output` | Output directory where models and results will be saved. | Required |

### Output files

Both scripts write results to the directory specified by `-output`. Typical outputs include:

| File | Description |
|---|---|
| `trained_model.sav` | Serialized model (joblib) — reusable for prediction on new sequences |
| `training_kfold(10)_metrics.csv` | 10-fold cross-validation metrics on the training set |
| `training_confusion_matrix.csv` | Confusion matrix for the training set (classification only) |
| `metrics_test.csv` | Evaluation metrics on the held-out test set |
| `test_confusion_matrix.csv` | Confusion matrix for the test set (classification only) |
| `test_predictions.csv` | Per-sequence predictions on the test set |
| `feature_importance.tsv` | Feature importance scores |
| `best_descriptors/` | Best-selected descriptor matrices for train and test sets |

## Trained Models

The platform hosts a continuously expanding repository of pre-trained, benchmarked models for genomic, transcriptomic, and proteomic applications. You can browse and use these models directly through the web platform at https://bioautoml.icmc.usp.br/.

To download all trained models for offline use, they are available on Zenodo:

**[https://doi.org/10.5281/zenodo.20349210](https://doi.org/10.5281/zenodo.20349210)**

## Citation

If you use this code in a scientific publication, we would appreciate citations to the following paper:

Silva de Almeida, B. L., Bonidia, R., Bole, M., Avila-Santos, A., Stadler, P. F., Nunes da Rocha, U., & de Carvalho, A. C. L. F. (2026). BioAutoML-FAST: an automated machine-learning platform for reusable and benchmarked biological sequence models. bioRxiv, 2026-04. [DOI](https://doi.org/10.64898/2026.04.18.719383)

```bibtex
@article{silva2026bioautoml,
  title={BioAutoML-FAST: an automated machine-learning platform for reusable and benchmarked biological sequence models},
  author={Silva de Almeida, Breno Livio and Bonidia, Robson and Bole, Martin and Avila-Santos, Anderson and Stadler, Peter F and Nunes da Rocha, Ulisses and de Carvalho, Andre CP L F},
  journal={bioRxiv},
  pages={2026--04},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```