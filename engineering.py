import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import polars as pl
import argparse
from homology import get_homology_folds, prepare_homology, add_homology_arguments, resolve_homology_arguments
import subprocess
import shutil
import sys
import json
import os.path
import random
import time
import lightgbm as lgb
import optuna
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, roc_auc_score, matthews_corrcoef, average_precision_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from Bio import SeqIO
from feature_execution import get_available_cpus, run_feature_commands


NUCLEOTIDE_DESCRIPTORS = [
	'NAC', 'DNC', 'TNC', 'kGap_di', 'kGap_tri', 'ORF', 'Fickett',
	'Shannon', 'FourierBinary', 'FourierComplex', 'Tsallis', 'Revkmer',
	'PseDNC', 'PseKNC', 'SC-PseDNC', 'SC-PseTNC', 'DAC', 'TAC', 'TCC',
	'TACC'
]

AMINOACID_DESCRIPTORS = [
	'Shannon', 'Tsallis_23', 'Tsallis_30', 'Tsallis_40',
	'ComplexNetworks', 'kGap', 'AAC', 'DPC', 'CKSAAP', 'DDE', 'GAAC',
	'CKSAAGP', 'GDPC', 'GTPC', 'CTDC', 'CTDT', 'CTDD', 'CTriad',
	'KSCTriad', 'Global', 'Peptide', 'Fourier_Integer', 'Fourier_EIIP'
]

REPDNA_DESCRIPTORS = {
	'Revkmer', 'PseDNC', 'PseKNC', 'SC-PseDNC', 'SC-PseTNC',
	'DAC', 'TAC', 'TCC', 'TACC'
}

SELF_PREFIXED_DATASETS = {'iFeature-features', 'repDNA'}

# The publication-compatible schema stores every descriptor exactly as
# originally generated, including overlapping blocks such as CKSAAP gap 0.
DESCRIPTOR_FEATURE_DEPENDENCIES = {}

random_seed = 63
optimization_seed = 63


def prepare_fasta_inputs(files, labels):
	"""Sort paired FASTA inputs without separating files from their labels."""
	if not files or not labels or len(files) != len(labels):
		raise ValueError('Provide exactly one label for every FASTA file.')
	if len(set(map(os.path.abspath, files))) != len(files):
		raise ValueError('The same FASTA file was supplied more than once.')
	if len(set(map(os.path.basename, files))) != len(files):
		raise ValueError('FASTA basenames must be unique within each split.')
	pairs = sorted(zip(files, labels), key=lambda pair: (str(pair[1]), os.path.abspath(pair[0])))
	return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def descriptor_prefix(descriptor):
	"""Return the schema prefix used by one Stage 1 descriptor group."""

	if descriptor in REPDNA_DESCRIPTORS:
		return f'repDNA__{descriptor}__'
	return f'{descriptor}__'


def get_descriptor_indices(columns, descriptors, require_all=False):
	"""Map descriptor groups to feature positions using schema-v2 names."""

	column_names = list(columns)
	standalone_indices = {}
	for descriptor in descriptors:
		prefix = descriptor_prefix(descriptor)
		standalone_indices[descriptor] = [
			index for index, column in enumerate(column_names)
			if column.startswith(prefix)
		]
		if require_all and not standalone_indices[descriptor]:
			raise ValueError(
				f'No columns found for descriptor {descriptor!r} with prefix {prefix!r}.'
			)

	indices = {}
	for descriptor in descriptors:
		indices[descriptor] = []
		for dependency in DESCRIPTOR_FEATURE_DEPENDENCIES.get(descriptor, ()):
			indices[descriptor].extend(standalone_indices.get(dependency, []))
		indices[descriptor].extend(standalone_indices[descriptor])
	return indices


def get_selected_feature_indices(descriptor_indices, descriptor_presence):
	"""Return the ordered union of all selected logical descriptor groups."""

	selected = []
	seen = set()
	for descriptor, indices in descriptor_indices.items():
		if not descriptor_presence[descriptor]:
			continue
		for index in indices:
			if index not in seen:
				selected.append(index)
				seen.add(index)
	return selected


def expand_descriptor_dependencies(descriptors):
	"""Add canonical feature blocks required by selected logical groups."""

	expanded = list(descriptors)
	for descriptor in descriptors:
		for dependency in DESCRIPTOR_FEATURE_DEPENDENCIES.get(descriptor, ()):
			if dependency not in expanded:
				expanded.append(dependency)
	return expanded


def remove_constant_features(dataframe, output_path):
	"""Remove all-missing and train-constant columns and record the fitted mask."""

	variable = dataframe.nunique(dropna=True) > 1
	removed = dataframe.columns[~variable]
	os.makedirs(output_path, exist_ok=True)
	pd.DataFrame({'feature': removed}).to_csv(
		os.path.join(output_path, 'removed_constant_features.csv'), index=False
	)
	return dataframe.loc[:, variable]


def feature_dataset_name(dataset):
	return os.path.splitext(os.path.basename(dataset))[0]


def read_feature_dataset(dataset):
	"""Read one extractor output and apply the schema-v2 group prefix."""

	descriptor = feature_dataset_name(dataset)
	dataframe = pl.read_csv(dataset, infer_schema=False)
	dataframe = dataframe.filter(pl.col('nameseq') != 'nameseq')
	if dataframe['nameseq'].n_unique() != dataframe.height:
		raise ValueError(f'Duplicate sequence identifiers in {dataset}.')

	if descriptor not in SELF_PREFIXED_DATASETS:
		rename = {
			column: f'{descriptor}__{column}'
			for column in dataframe.columns
			if column not in {'nameseq', 'label'}
		}
		dataframe = dataframe.rename(rename)
	return dataframe.select(pl.all().exclude('nameseq'), pl.col('nameseq')).sort('nameseq')


def combine_feature_datasets(datasets):
	"""Align descriptor outputs by sequence ID and validate their row sets."""

	dataframes = [read_feature_dataset(dataset) for dataset in datasets]
	expected_ids = set(dataframes[0]['nameseq'].to_list())
	expected_labels = dict(zip(
		dataframes[0]['nameseq'].to_list(), dataframes[0]['label'].to_list()
	))
	for dataset, dataframe in zip(datasets[1:], dataframes[1:]):
		observed_ids = set(dataframe['nameseq'].to_list())
		if observed_ids != expected_ids:
			missing = sorted(expected_ids - observed_ids)[:5]
			extra = sorted(observed_ids - expected_ids)[:5]
			raise ValueError(
				f'Sequence IDs do not align in {dataset}; missing={missing}, extra={extra}.'
			)
		observed_labels = dict(zip(
			dataframe['nameseq'].to_list(), dataframe['label'].to_list()
		))
		if observed_labels != expected_labels:
			raise ValueError(f'Sequence labels do not align in {dataset}.')

	combined = pl.concat(dataframes, how='align')
	feature_columns = [
		column for column in combined.columns
		if column not in {'nameseq', 'label'}
	]
	if len(feature_columns) != len(set(feature_columns)):
		raise ValueError('Feature schema contains duplicate column names.')
	return combined


def save_feature_schema(path, data_type, feature_columns):
	"""Persist the exact feature contract used by training and prediction."""

	schema = {
		'version': 1,
		'data_type': data_type,
		'feature_count': len(feature_columns),
		'features': list(feature_columns),
	}
	with open(os.path.join(path, 'feature_schema.json'), 'w') as schema_file:
		json.dump(schema, schema_file, indent=2)

class EarlyStoppingCallback:
	"""Optuna callback that stops a study when no meaningful improvement is seen for `patience` consecutive trials."""

	def __init__(self, patience, min_delta):
		"""patience: number of consecutive trials without improvement before stopping.
		min_delta: minimum absolute change in best value that counts as an improvement.
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.best_value = None
		self.no_improve_count = 0

	def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
		"""Called after each trial; calls study.stop() once patience is exhausted."""
		# Ignore pruned or failed trials
		if trial.state != optuna.trial.TrialState.COMPLETE:
			return

		current_best = study.best_value

		if self.best_value is None:
			self.best_value = current_best
			return

		# Check if improvement is meaningful
		if np.abs(current_best - self.best_value) >= self.min_delta:
			self.best_value = current_best
			self.no_improve_count = 0
		else:
			self.no_improve_count += 1

		if self.no_improve_count >= self.patience:
			print(
				f"Early stopping triggered: "
				f"no improvement ≥ {self.min_delta} "
				f"in {self.patience} trials."
			)
			study.stop()

def get_cpu_parameters(n_cpu, requested_search_jobs=None):
	"""Return total CPUs, Optuna workers, and threads available to each trial."""

	total_cpus = get_available_cpus(n_cpu)
	if requested_search_jobs is None:
		search_jobs = min(16, total_cpus)
	else:
		search_jobs = min(max(1, requested_search_jobs), total_cpus)
	model_jobs = max(1, total_cpus // search_jobs)
	return total_cpus, search_jobs, model_jobs

def get_available_memory():
	"""Return available physical memory in bytes when the platform exposes it."""

	try:
		return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
	except (AttributeError, OSError, ValueError):
		return None

def prepare_feature_cache(train, y, task, folds=None):
	"""Cache fold-wise mean-imputed matrices when enough memory is available."""

	if task == 0:
		cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
	else:
		cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)

	if folds is None:
		folds = list(cv.split(train, y))
	estimated_bytes = train.to_numpy(copy=False).nbytes * len(folds)
	available_memory = get_available_memory()
	if available_memory is not None and estimated_bytes > available_memory * 0.25:
		print('Stage 1 fold cache disabled: estimated cache exceeds 25% of available memory.')
		return None, folds

	for train_idx, _ in folds:
		if train.iloc[train_idx].isna().all(axis=0).any():
			print('Stage 1 fold cache disabled: a training fold contains an all-missing feature.')
			return None, folds

	feature_cache = []
	for train_idx, val_idx in folds:
		imputer = SimpleImputer(strategy="mean")
		X_train = imputer.fit_transform(train.iloc[train_idx])
		X_val = imputer.transform(train.iloc[val_idx])
		feature_cache.append((X_train, X_val, train_idx, val_idx))

	return feature_cache, folds

def objective_nucleotide(trial, train, task, y, feature_cache=None, folds=None, model_jobs=None):
	"""Automated Feature Engineering - Optuna - Objective Function - Bayesian Optimization"""

	# Define search space
	space = {
		descriptor: trial.suggest_categorical(descriptor, [0, 1])
		for descriptor in NUCLEOTIDE_DESCRIPTORS
	}

	# Descriptor indices
	descriptors = get_descriptor_indices(train.columns, NUCLEOTIDE_DESCRIPTORS)
	index = get_selected_feature_indices(descriptors, space)

	if len(index) == 0:
		raise optuna.TrialPruned()

	# === Task Handling ===
	if task == 0:
		model = Pipeline([
			("imputer", SimpleImputer(strategy="mean")),
			("clf", lgb.LGBMClassifier(
				random_state=random_seed,
				verbosity=-1,
				n_jobs=model_jobs
			))
		])
		cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
	elif task == 1:
		model = Pipeline([
			("imputer", SimpleImputer(strategy="mean")),
			("reg", lgb.LGBMRegressor(
				random_state=random_seed,
				verbosity=-1,
				n_jobs=model_jobs
			))
		])
		cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
	else:
		raise ValueError("Invalid task. Use 0 (classification) or 1 (regression).")

	if isinstance(y, list):
		y = np.array(y)

	# === Cross-Validation ===
	fold_scores = []
	X_subset = train.iloc[:, index] if feature_cache is None else None
	folds = folds if folds is not None else list(cv.split(X_subset, y))

	try:
		# Manual CV loop to enable pruning
		for step, (train_idx, val_idx) in enumerate(folds):
			if feature_cache is None:
				X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
				fold_model = model
			else:
				cached_train, cached_val, _, _ = feature_cache[step]
				X_train, X_val = cached_train[:, index], cached_val[:, index]
				fold_model = model.named_steps["clf"] if task == 0 else model.named_steps["reg"]
			y_train, y_val = y[train_idx], y[val_idx]
			
			fold_model.fit(X_train, y_train)
			preds = fold_model.predict(X_val)
			
			if task == 0:
				# MCC for classification
				val_score = matthews_corrcoef(y_val, preds)
			else:
				# RMSE for regression
				val_score = root_mean_squared_error(y_val, preds)
			
			fold_scores.append(val_score)
			
			# Report the mean score so far to Optuna
			intermediate_value = np.mean(fold_scores)
			trial.report(intermediate_value, step)
			
			# Prune if this trial is performing poorly
			if trial.should_prune():
				raise optuna.TrialPruned()
		
		metric = np.mean(fold_scores)
	except optuna.TrialPruned:
		raise
	except Exception as e:
		print("Trial failed with exception:")
		print(type(e).__name__, str(e))
		raise optuna.TrialPruned()

	return metric

def feature_engineering_nucleotide(task, estimations, fnameseqtrain, train, train_labels, test, foutput, n_cpu=-1, search_jobs=None, homology_report=None):
	"""Select the best subset of nucleotide descriptors via Bayesian optimization (Optuna TPE).

	Treats each descriptor group as a binary on/off variable and maximises MCC (task=0)
	or minimises RMSE (task=1) using 5-fold CV with LightGBM.
	Saves selected_descriptors.csv, best_train.csv, and best_test.csv under foutput/best_descriptors/.
	Returns (path_btrain, path_btest, btrain_df, btest_df).
	"""
	print('Automated Feature Engineering - Bayesian Optimization')

	df_x = pd.read_csv(train)
	df_x = df_x.replace([np.inf, -np.inf], np.nan)
	get_descriptor_indices(df_x.columns, NUCLEOTIDE_DESCRIPTORS, require_all=True)
	
	path_bio = foutput + '/best_descriptors'
	if not os.path.exists(path_bio):
		os.mkdir(path_bio)

	param = {descriptor: [0, 1] for descriptor in NUCLEOTIDE_DESCRIPTORS}
	
	if task == 0:
		labels = pd.read_csv(train_labels)
		le = LabelEncoder()
		y = le.fit_transform(labels)
		direction = "maximize"
	elif task == 1:
		y = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(fnameseqtrain)["nameseq"].to_list()]
		direction = "minimize"

	_, search_jobs, model_jobs = get_cpu_parameters(n_cpu, search_jobs)
	folds = get_homology_folds(homology_report, pd.read_csv(fnameseqtrain).values, task, random_seed) if homology_report else None
	feature_cache, folds = prepare_feature_cache(df_x, y, task, folds)
	func = lambda trial: objective_nucleotide(trial, df_x, task, y, feature_cache, folds, model_jobs)
	
	early_stopping = EarlyStoppingCallback(
		patience=patience,
		min_delta=difference
	)

	results = optuna.create_study(
		direction=direction,
		sampler=optuna.samplers.TPESampler(
			n_startup_trials=30, multivariate=True, group=True,
			constant_liar=True, seed=optimization_seed
		)
	)

	results.optimize(
		func,
		n_trials=estimations,
		timeout=10_800,
		show_progress_bar=True,
		callbacks=[early_stopping],
		n_jobs=search_jobs
	)

	best_tuning = results.best_params
	print(best_tuning)
	
	descriptors = get_descriptor_indices(df_x.columns, NUCLEOTIDE_DESCRIPTORS)

	# Get indices of selected descriptors
	descriptor_presence = {}
	for descriptor in descriptors:
		result = best_tuning[descriptor]
		if result == 1:
			descriptor_presence[descriptor] = 1
		else:
			descriptor_presence[descriptor] = 0
	index = get_selected_feature_indices(descriptors, descriptor_presence)

	# Save presence/absence table
	df_presence = pd.DataFrame([descriptor_presence])
	df_presence.to_csv(os.path.join(path_bio, 'selected_descriptors.csv'), index=False)
	
	selected_columns = df_x.columns[index]
	btrain = df_x.iloc[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.to_csv(path_btrain, index=False)

	if test != '':
		btest = pd.read_csv(test, usecols=selected_columns).loc[:, selected_columns]
		path_btest = path_bio + '/best_test.csv'
		btest.to_csv(path_btest, index=False)
	else:
		btest, path_btest = '', ''

	return path_btrain, path_btest, btrain, btest

def objective_aminoacid(trial, train, task, y, feature_cache=None, folds=None, model_jobs=None):

	"""Automated Feature Engineering - Optuna - Objective Function - Bayesian Optimization"""

	space = {
		descriptor: trial.suggest_categorical(descriptor, [0, 1])
		for descriptor in AMINOACID_DESCRIPTORS
	}

	descriptors = get_descriptor_indices(train.columns, AMINOACID_DESCRIPTORS)
	index = get_selected_feature_indices(descriptors, space)

	if len(index) == 0:
		raise optuna.TrialPruned()

	# === Task Handling ===
	if task == 0:
		model = Pipeline([
			("imputer", SimpleImputer(strategy="mean")),
			("clf", lgb.LGBMClassifier(
				random_state=random_seed,
				verbosity=-1,
				n_jobs=model_jobs
			))
		])
		cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
	elif task == 1:
		model = Pipeline([
			("imputer", SimpleImputer(strategy="mean")),
			("reg", lgb.LGBMRegressor(
				random_state=random_seed,
				verbosity=-1,
				n_jobs=model_jobs
			))
		])
		cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
	else:
		raise ValueError("Invalid task. Use 0 (classification) or 1 (regression).")

	if isinstance(y, list):
		y = np.array(y)
		
	fold_scores = []
	X_subset = train.iloc[:, index] if feature_cache is None else None
	folds = folds if folds is not None else list(cv.split(X_subset, y))

	try:
		for step, (train_idx, val_idx) in enumerate(folds):
			if feature_cache is None:
				X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
				fold_model = model
			else:
				cached_train, cached_val, _, _ = feature_cache[step]
				X_train, X_val = cached_train[:, index], cached_val[:, index]
				fold_model = model.named_steps["clf"] if task == 0 else model.named_steps["reg"]
			y_train, y_val = y[train_idx], y[val_idx]

			fold_model.fit(X_train, y_train)
			preds = fold_model.predict(X_val)

			if task == 0:
				val_score = matthews_corrcoef(y_val, preds)
			else:
				val_score = root_mean_squared_error(y_val, preds)
			
			fold_scores.append(val_score)

			# Report mean to Optuna for pruning
			trial.report(np.mean(fold_scores), step)

			if trial.should_prune():
				raise optuna.TrialPruned()
		
		metric = np.mean(fold_scores)

	except optuna.TrialPruned:
		raise
	except Exception as e:
		print("Trial failed with exception:")
		print(type(e).__name__, str(e))
		raise optuna.TrialPruned()
		
	return metric

def feature_engineering_aminoacid(task, estimations, fnameseqtrain, train, train_labels, test, foutput, n_cpu=-1, search_jobs=None, homology_report=None):
	"""Select the best subset of amino acid descriptors via Bayesian optimization (Optuna TPE).

	Treats each descriptor group as a binary on/off variable and maximises MCC (task=0)
	or minimises RMSE (task=1) using 5-fold CV with LightGBM.
	Saves selected_descriptors.csv, best_train.csv, and best_test.csv under foutput/best_descriptors/.
	Returns (path_btrain, path_btest, btrain_df, btest_df).
	"""

	print('Automated Feature Engineering - Bayesian Optimization')

	df_x = pd.read_csv(train)
	df_x = df_x.replace([np.inf, -np.inf], np.nan)
	get_descriptor_indices(df_x.columns, AMINOACID_DESCRIPTORS, require_all=True)

	path_bio = foutput + '/best_descriptors'
	if not os.path.exists(path_bio):
		os.mkdir(path_bio)

	param = {descriptor: [0, 1] for descriptor in AMINOACID_DESCRIPTORS}

	if task == 0:
		labels = pd.read_csv(train_labels)
		le = LabelEncoder()
		y = le.fit_transform(labels)
		direction = "maximize"
	elif task == 1:
		y = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(fnameseqtrain)["nameseq"].to_list()]
		direction = "minimize"

	_, search_jobs, model_jobs = get_cpu_parameters(n_cpu, search_jobs)
	folds = get_homology_folds(homology_report, pd.read_csv(fnameseqtrain).values, task, random_seed) if homology_report else None
	feature_cache, folds = prepare_feature_cache(df_x, y, task, folds)
	func = lambda trial: objective_aminoacid(trial, df_x, task, y, feature_cache, folds, model_jobs)

	early_stopping = EarlyStoppingCallback(
		patience=patience,
		min_delta=difference
	)

	results = optuna.create_study(
		direction=direction,
		sampler=optuna.samplers.TPESampler(
			n_startup_trials=30, multivariate=True, group=True,
			constant_liar=True, seed=optimization_seed
		)
	)

	results.optimize(
		func,
		n_trials=estimations,
		timeout=7200,
		show_progress_bar=True,
		callbacks=[early_stopping],
		n_jobs=search_jobs
	)

	best_tuning = results.best_params
	print(best_tuning)
	
	descriptors = get_descriptor_indices(df_x.columns, AMINOACID_DESCRIPTORS)

	# Determine which descriptors were selected
	descriptor_presence = {}
	for descriptor in descriptors:
		result = best_tuning[descriptor]
		if result == 1:
			descriptor_presence[descriptor] = 1
		else:
			descriptor_presence[descriptor] = 0
	index = get_selected_feature_indices(descriptors, descriptor_presence)

	# Save presence/absence summary CSV
	df_presence = pd.DataFrame([descriptor_presence])
	df_presence.to_csv(os.path.join(path_bio, 'selected_descriptors.csv'), index=False)

	selected_columns = df_x.columns[index]
	btrain = df_x.iloc[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.to_csv(path_btrain, index=False)

	if test != '':
		btest = pd.read_csv(test, usecols=selected_columns).loc[:, selected_columns]
		path_btest = path_bio + '/best_test.csv'
		btest.to_csv(path_btest, index=False)
	else:
		btest, path_btest = '', ''

	return path_btrain, path_btest, btrain, btest

def feature_extraction_aminoacid(ftrain, ftrain_labels, ftest, ftest_labels, foutput, n_cpu=-1):
	"""Extract amino acid descriptors from FASTA files and concatenate them into train/test CSVs.

	Runs all feature extractors (Shannon, Tsallis, ComplexNetworks, kGap, AAC, DPC, iFeature,
	modlAMP Global/Peptide, Fourier Integer/EIIP) in parallel subprocesses.
	Aligns all descriptor CSVs by sequence name using Polars, then splits by train/test membership.
	Writes fnameseqtrain, ftrain, flabeltrain (and test equivalents) under foutput/feat_extraction/.
	Returns (fnameseqtrain, fnameseqtest, ftrain, flabeltrain, ftest, flabeltest) as file paths.
	"""

	# Setup directories
	path = os.path.join(foutput, 'feat_extraction')
	path_results = foutput

	# Clear and create directories
	for dir_path in [path_results, path]:
		# try:
		# 	shutil.rmtree(dir_path)
		# except OSError:
		# 	pass
		os.makedirs(dir_path, exist_ok=True)

	# Create train/test subdirectories
	for subdir in ['train', 'test']:
		os.makedirs(os.path.join(path, subdir), exist_ok=True)

	# Organize input files
	input_groups = [
		(ftrain, ftrain_labels, 'train'),
		(ftest, ftest_labels, 'test') if ftest else (None, None, None)
	]
	input_groups = [x for x in input_groups if x[0] is not None]

	sequence_train = set()
	fasta_list = []
	datasets = [
		'Shannon.csv',
		'Tsallis_23.csv',
		'Tsallis_30.csv',
		'Tsallis_40.csv',
		'ComplexNetworks.csv',
		'kGap.csv',
		'AAC.csv',
		'DPC.csv',
		'iFeature-features.csv',
		'Global.csv',
		'Peptide.csv'
	]

	datasets = [os.path.join(path, fname) for fname in datasets]
	for dataset in datasets:
		if os.path.exists(dataset):
			os.remove(dataset)

	print('Extracting features...')

	for fasta_files, label_files, split_type in input_groups:
		for fasta_file, label_file in zip(fasta_files, label_files):
			# Preprocess file
			file_name = os.path.basename(fasta_file)
			preprocessed_fasta = os.path.join(path, split_type, f'pre_{file_name}')
			
			subprocess.run([
				sys.executable, 'other-methods/preprocessing.py',
				'-i', fasta_file,
				'-o', preprocessed_fasta,
				'-s', f'{split_type}_{label_file}',
				'-d', "Protein",
			], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
			
			if split_type == 'train':
				with open(preprocessed_fasta) as handle:
					sequence_train.update(str(record.id) for record in SeqIO.parse(handle, "fasta"))
			
			fasta_list.append(preprocessed_fasta)
			
			# Define all feature extraction commands
			commands = [
				[sys.executable, 'other-methods/EntropyClass.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'Shannon.csv'),
				'-l', label_file, '-k', '5', '-e', 'Shannon'],

				[sys.executable, 'other-methods/TsallisEntropy.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'Tsallis_23.csv'),
				'-l', label_file, '-k', '5', '-q', '2.3'],
				
				[sys.executable, 'other-methods/TsallisEntropy.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'Tsallis_30.csv'),
				'-l', label_file, '-k', '5', '-q', '3.0'],
				
				[sys.executable, 'other-methods/TsallisEntropy.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'Tsallis_40.csv'),
				'-l', label_file, '-k', '5', '-q', '4.0'],
				
				[sys.executable, 'other-methods/mathfeature-modified/methods/ComplexNetworksClass-v2.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'ComplexNetworks.csv'),
				'-l', label_file, '-k', '3'],
				
				[sys.executable, 'MathFeature/methods/Kgap.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'kGap.csv'),
				'-l', label_file, '-k', '1', '-bef', '1', '-aft', '1', '-seq', '3'],
				
				[sys.executable, 'other-methods/ExtractionTechniques-Protein.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'AAC.csv'),
				'-l', label_file, '-t', 'AAC'],
				
				[sys.executable, 'other-methods/ExtractionTechniques-Protein.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'DPC.csv'),
				'-l', label_file, '-t', 'DPC'],
				
				[sys.executable, 'other-methods/iFeature-modified/iFeature.py',
				'--file', preprocessed_fasta, '--type', 'All',
				'--label', label_file, '--out', os.path.join(path, 'iFeature-features.csv')],
				
				[sys.executable, 'other-methods/modlAMP-modified/descriptors.py',
				'-option', 'global', '-label', label_file,
				'-input', preprocessed_fasta, '-output', os.path.join(path, 'Global.csv')],
				
				[sys.executable, 'other-methods/modlAMP-modified/descriptors.py',
				'-option', 'peptide', '-label', label_file,
				'-input', preprocessed_fasta, '-output', os.path.join(path, 'Peptide.csv')]
			]
			
			log_dir = os.path.join(path, 'logs')
			os.makedirs(log_dir, exist_ok=True)  # make sure the folder exists

			run_feature_commands([
				(feature_dataset_name(dataset), command)
				for command, dataset in zip(commands, datasets)
			], log_dir, n_cpu)

	# Process Fourier features
	labels_list = ftrain_labels + (ftest_labels if ftest else [])
	text_input = '\n'.join(f'{fasta}\n{label}' for fasta, label in zip(fasta_list, labels_list))

	fourier_datasets = [
		('Fourier_Integer.csv', '6'),
		('Fourier_EIIP.csv', '8')
	]

	for fname, r_val in fourier_datasets:
		dataset = os.path.join(path, fname)
		if os.path.exists(dataset):
			os.remove(dataset)
		subprocess.run([
			sys.executable, 'other-methods/mathfeature-modified/methods/Mappings-Protein.py',
			'-n', str(len(fasta_list)), '-o', dataset, '-r', r_val
		], text=True, input=text_input, stdout=subprocess.DEVNULL,
			stderr=subprocess.STDOUT, check=True)
		datasets.append(dataset)

	"""Concatenating all the extracted features"""
	
	if datasets:
		dataframes = combine_feature_datasets(datasets)

		dataframes = dataframes.with_columns(
			pl.when(pl.col("nameseq").is_in(sequence_train))
			.then(pl.lit("train"))
			.otherwise(pl.lit("test"))
			.alias("split_type")
		)

	X_train = dataframes.filter(pl.col("split_type") == "train")

	nameseq_train = X_train.select("nameseq")
	fnameseqtrain = os.path.join(path, "fnameseqtrain.csv")
	nameseq_train.write_csv(fnameseqtrain)

	y_train = X_train.select("label")
	flabeltrain = os.path.join(path, "flabeltrain.csv")
	y_train.write_csv(flabeltrain)

	ftrain = os.path.join(path, "ftrain.csv")
	train_features = X_train.select(pl.all().exclude(["split_type", "nameseq", "label"]))
	train_features.write_csv(ftrain)
	save_feature_schema(path, 'Protein', train_features.columns)
	
	fnameseqtest, ftest, flabeltest = '', '', ''

	if ftest:
		X_test = dataframes.filter(pl.col("split_type") == "test")

		nameseq_test = X_test.select("nameseq")
		fnameseqtest = os.path.join(path, "fnameseqtest.csv")
		nameseq_test.write_csv(fnameseqtest)

		y_test = X_test.select("label")
		flabeltest = os.path.join(path, "flabeltest.csv")
		y_test.write_csv(flabeltest)

		ftest = os.path.join(path, "ftest.csv")
		X_test.select(pl.all().exclude(["split_type", "nameseq", "label"])).write_csv(ftest)

	return fnameseqtrain, fnameseqtest, ftrain, flabeltrain, ftest, flabeltest

def feature_extraction_nucleotide(ftrain, ftrain_labels, ftest, ftest_labels, foutput, n_cpu=-1):
	"""Extract nucleotide descriptors from FASTA files and concatenate them into train/test CSVs.

	Runs all feature extractors (NAC, DNC, TNC, kGap, ORF, Fickett, Shannon, Fourier Binary/Complex,
	Tsallis, repDNA) in parallel subprocesses.
	Aligns all descriptor CSVs by sequence name using Polars, then splits by train/test membership.
	Writes fnameseqtrain, ftrain, flabeltrain (and test equivalents) under foutput/feat_extraction/.
	Returns (fnameseqtrain, fnameseqtest, ftrain, flabeltrain, ftest, flabeltest) as file paths.
	"""

	# Setup directories
	path = os.path.join(foutput, 'feat_extraction')
	path_results = foutput

	# Clear and create directories
	for dir_path in [path_results, path]:
		os.makedirs(dir_path, exist_ok=True)

	# Create train/test subdirectories
	for subdir in ['train', 'test']:
		os.makedirs(os.path.join(path, subdir), exist_ok=True)

	# Organize input files
	input_groups = [
		(ftrain, ftrain_labels, 'train'),
		(ftest, ftest_labels, 'test') if ftest else (None, None, None)
	]
	input_groups = [x for x in input_groups if x[0] is not None]

	sequence_train = set()
	fasta_list = []
	datasets = [
		'NAC.csv',
		'DNC.csv',
		'TNC.csv',
		'kGap_di.csv',
		'kGap_tri.csv',
		'ORF.csv',
		'Fickett.csv',
		'Shannon.csv',
		'FourierBinary.csv',
		'FourierComplex.csv',
		'Tsallis.csv',
		'repDNA.csv'
	]

	datasets = [os.path.join(path, fname) for fname in datasets]
	for dataset in datasets:
		if os.path.exists(dataset):
			os.remove(dataset)

	print('Extracting features...')

	for fasta_files, label_files, split_type in input_groups:
		for fasta_file, label_file in zip(fasta_files, label_files):
			# Preprocess file
			file_name = os.path.basename(fasta_file)
			preprocessed_fasta = os.path.join(path, split_type, f'pre_{file_name}')
			
			subprocess.run([
				sys.executable, 'other-methods/preprocessing.py',
				'-i', fasta_file,
				'-o', preprocessed_fasta,
				'-s', f'{split_type}_{label_file}',
				'-d', "DNA/RNA",
			], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
			
			if split_type == 'train':
				with open(preprocessed_fasta) as handle:
					sequence_train.update(str(record.id) for record in SeqIO.parse(handle, "fasta"))
			
			fasta_list.append(preprocessed_fasta)
			
			# Define all feature extraction commands
			commands = [
				[sys.executable, 'MathFeature/methods/ExtractionTechniques.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'NAC.csv'), '-l', label_file,
				'-t', 'NAC', '-seq', '1'],

				[sys.executable, 'MathFeature/methods/ExtractionTechniques.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'DNC.csv'), '-l', label_file,
				'-t', 'DNC', '-seq', '1'],

				[sys.executable, 'MathFeature/methods/ExtractionTechniques.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'TNC.csv'), '-l', label_file,
				'-t', 'TNC', '-seq', '1'],

				[sys.executable, 'MathFeature/methods/Kgap.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'kGap_di.csv'), '-l',
				label_file, '-k', '1', '-bef', '1', '-aft', '2', '-seq', '1'],

				[sys.executable, 'MathFeature/methods/Kgap.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'kGap_tri.csv'), '-l',
				label_file, '-k', '1', '-bef', '1', '-aft', '3', '-seq', '1'],

				[sys.executable, 'MathFeature/methods/CodingClass.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'ORF.csv'), '-l', label_file],

				[sys.executable, 'other-methods/mathfeature-modified/methods/FickettScore.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'Fickett.csv'), '-l', label_file,
				'-seq', '1'],

				[sys.executable, 'MathFeature/methods/EntropyClass.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'Shannon.csv'), '-l', label_file,
				'-k', '5', '-e', 'Shannon'],

				[sys.executable, 'other-methods/mathfeature-modified/methods/FourierClass.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'FourierBinary.csv'), '-l', label_file,
				'-r', '1'],

				[sys.executable, 'other-methods/FourierClass.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'FourierComplex.csv'), '-l', label_file,
				'-r', '6'],

				[sys.executable, 'other-methods/TsallisEntropy.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'Tsallis.csv'), '-l', label_file,
				'-k', '5', '-q', '2.3'],

				[sys.executable, 'other-methods/repDNA/repDNA-feat.py', '--file',
				preprocessed_fasta, '--output', os.path.join(path, 'repDNA.csv'), '--label', label_file]
			]
			
			log_dir = os.path.join(path, 'logs')
			os.makedirs(log_dir, exist_ok=True)  # make sure the folder exists

			run_feature_commands([
				(feature_dataset_name(dataset), command)
				for command, dataset in zip(commands, datasets)
			], log_dir, n_cpu)

	"""Concatenating all the extracted features"""
	
	if datasets:
		dataframes = combine_feature_datasets(datasets)
		
		dataframes = dataframes.with_columns(
			pl.when(pl.col("nameseq").is_in(sequence_train))
			.then(pl.lit("train"))
			.otherwise(pl.lit("test"))
			.alias("split_type")
		)

	X_train = dataframes.filter(pl.col("split_type") == "train")

	nameseq_train = X_train.select("nameseq")
	fnameseqtrain = os.path.join(path, "fnameseqtrain.csv")
	nameseq_train.write_csv(fnameseqtrain)

	y_train = X_train.select("label")
	flabeltrain = os.path.join(path, "flabeltrain.csv")
	y_train.write_csv(flabeltrain)

	ftrain = os.path.join(path, "ftrain.csv")
	train_features = X_train.select(pl.all().exclude(["split_type", "nameseq", "label"]))
	train_features.write_csv(ftrain)
	save_feature_schema(path, 'DNA/RNA', train_features.columns)
	
	fnameseqtest, ftest, flabeltest = '', '', ''

	if ftest:
		X_test = dataframes.filter(pl.col("split_type") == "test")

		nameseq_test = X_test.select("nameseq")
		fnameseqtest = os.path.join(path, "fnameseqtest.csv")
		nameseq_test.write_csv(fnameseqtest)

		y_test = X_test.select("label")
		flabeltest = os.path.join(path, "flabeltest.csv")
		y_test.write_csv(flabeltest)

		ftest = os.path.join(path, "ftest.csv")
		X_test.select(pl.all().exclude(["split_type", "nameseq", "label"])).write_csv(ftest)

	return fnameseqtrain, fnameseqtest, ftrain, flabeltrain, ftest, flabeltest

def get_selected_descriptors(selected_descriptors):
	"""Read the descriptor names selected by Stage 1 in their stored order."""

	descriptors = pd.read_csv(selected_descriptors).iloc[0]
	return [name for name, selected in descriptors.items() if int(selected) == 1]

def prepare_test_fastas(fasta_test, fasta_label_test, data_type, path):
	"""Preprocess test FASTA files and return their paths and labels."""

	preprocessed_fastas = []
	for fasta_file, label_file in zip(fasta_test, fasta_label_test):
		file_name = os.path.basename(fasta_file)
		preprocessed_fasta = os.path.join(path, f'pre_{file_name}')
		subprocess.run([
			sys.executable, 'other-methods/preprocessing.py',
			'-i', fasta_file,
			'-o', preprocessed_fasta,
			'-s', f'test_{label_file}',
			'-d', data_type,
		], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
		preprocessed_fastas.append((preprocessed_fasta, label_file))

	return preprocessed_fastas

def save_selected_test(datasets, column_train, foutput):
	"""Align extracted test descriptors and save only the training columns."""

	dataframes = combine_feature_datasets(datasets).to_pandas()

	missing_columns = [column for column in column_train if column not in dataframes.columns]
	if missing_columns:
		raise ValueError(f'Missing selected test features: {missing_columns[:10]}')

	feat_path = os.path.join(foutput, 'feat_extraction')
	path_bio = os.path.join(foutput, 'best_descriptors')
	os.makedirs(path_bio, exist_ok=True)

	fnameseqtest = os.path.join(feat_path, 'fnameseqtest.csv')
	flabeltest = os.path.join(feat_path, 'flabeltest.csv')
	path_btest = os.path.join(path_bio, 'best_test.csv')

	dataframes[['nameseq']].to_csv(fnameseqtest, index=False)
	dataframes[['label']].to_csv(flabeltest, index=False)
	btest = dataframes.loc[:, list(column_train)]
	btest.to_csv(path_btest, index=False)

	return fnameseqtest, path_btest, flabeltest, btest

def feature_extraction_aminoacid_test(fasta_test, fasta_label_test, selected_descriptors, column_train, foutput, n_cpu=-1):
	"""Extract only Stage 1-selected amino-acid descriptors from test FASTA files."""

	selected = get_selected_descriptors(selected_descriptors)
	feat_path = os.path.join(foutput, 'feat_extraction')
	path = os.path.join(feat_path, 'test')
	selected_path = os.path.join(feat_path, 'selected_test_features')
	shutil.rmtree(path, ignore_errors=True)
	shutil.rmtree(selected_path, ignore_errors=True)
	os.makedirs(path, exist_ok=True)
	os.makedirs(selected_path, exist_ok=True)

	preprocessed_fastas = prepare_test_fastas(fasta_test, fasta_label_test, 'Protein', path)
	datasets = []
	ifeature_names = ['CKSAAP', 'DDE', 'GAAC', 'CKSAAGP', 'GDPC', 'GTPC',
					 'CTDC', 'CTDT', 'CTDD', 'CTriad', 'KSCTriad']
	selected_ifeature = [name for name in ifeature_names if name in selected]

	for preprocessed_fasta, label_file in preprocessed_fastas:
		commands = []
		command_map = {
			'Shannon': [sys.executable, 'other-methods/EntropyClass.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'Shannon.csv'), '-l', label_file, '-k', '5', '-e', 'Shannon'],
			'Tsallis_23': [sys.executable, 'other-methods/TsallisEntropy.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'Tsallis_23.csv'), '-l', label_file, '-k', '5', '-q', '2.3'],
			'Tsallis_30': [sys.executable, 'other-methods/TsallisEntropy.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'Tsallis_30.csv'), '-l', label_file, '-k', '5', '-q', '3.0'],
			'Tsallis_40': [sys.executable, 'other-methods/TsallisEntropy.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'Tsallis_40.csv'), '-l', label_file, '-k', '5', '-q', '4.0'],
			'ComplexNetworks': [sys.executable, 'other-methods/mathfeature-modified/methods/ComplexNetworksClass-v2.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'ComplexNetworks.csv'), '-l', label_file, '-k', '3'],
			'kGap': [sys.executable, 'MathFeature/methods/Kgap.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'kGap.csv'), '-l', label_file, '-k', '1', '-bef', '1', '-aft', '1', '-seq', '3'],
			'AAC': [sys.executable, 'other-methods/ExtractionTechniques-Protein.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'AAC.csv'), '-l', label_file, '-t', 'AAC'],
			'DPC': [sys.executable, 'other-methods/ExtractionTechniques-Protein.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'DPC.csv'), '-l', label_file, '-t', 'DPC'],
			'Global': [sys.executable, 'other-methods/modlAMP-modified/descriptors.py', '-option', 'global',
				'-label', label_file, '-input', preprocessed_fasta, '-output', os.path.join(selected_path, 'Global.csv')],
			'Peptide': [sys.executable, 'other-methods/modlAMP-modified/descriptors.py', '-option', 'peptide',
				'-label', label_file, '-input', preprocessed_fasta, '-output', os.path.join(selected_path, 'Peptide.csv')],
		}

		for descriptor, command in command_map.items():
			if descriptor in selected:
				commands.append((descriptor, command))
				dataset = os.path.join(selected_path, f'{descriptor}.csv')
				if dataset not in datasets:
					datasets.append(dataset)

		if selected_ifeature:
			dataset = os.path.join(selected_path, 'iFeature-features.csv')
			commands.append(('iFeature-features', [
				sys.executable, 'other-methods/iFeature-modified/iFeature.py', '--file', preprocessed_fasta,
				'--type', *selected_ifeature, '--label', label_file, '--out', dataset
			]))
			if dataset not in datasets:
				datasets.append(dataset)

		run_feature_commands(commands, os.path.join(selected_path, 'logs'), n_cpu)

	text_input = '\n'.join(f'{fasta}\n{label}' for fasta, label in preprocessed_fastas)
	for descriptor, file_name, representation in [
		('Fourier_Integer', 'Fourier_Integer.csv', '6'),
		('Fourier_EIIP', 'Fourier_EIIP.csv', '8'),
	]:
		if descriptor not in selected:
			continue
		dataset = os.path.join(selected_path, file_name)
		subprocess.run([
			sys.executable, 'other-methods/mathfeature-modified/methods/Mappings-Protein.py', '-n', str(len(preprocessed_fastas)),
			'-o', dataset, '-r', representation
		], text=True, input=text_input, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
		datasets.append(dataset)

	return save_selected_test(datasets, column_train, foutput)

def feature_extraction_nucleotide_test(fasta_test, fasta_label_test, selected_descriptors, column_train, foutput, n_cpu=-1):
	"""Extract only Stage 1-selected nucleotide descriptors from test FASTA files."""

	selected = get_selected_descriptors(selected_descriptors)
	feat_path = os.path.join(foutput, 'feat_extraction')
	path = os.path.join(feat_path, 'test')
	selected_path = os.path.join(feat_path, 'selected_test_features')
	shutil.rmtree(path, ignore_errors=True)
	shutil.rmtree(selected_path, ignore_errors=True)
	os.makedirs(path, exist_ok=True)
	os.makedirs(selected_path, exist_ok=True)

	preprocessed_fastas = prepare_test_fastas(fasta_test, fasta_label_test, 'DNA/RNA', path)
	datasets = []
	repdna_names = ['Revkmer', 'PseDNC', 'PseKNC', 'SC-PseDNC', 'SC-PseTNC',
					'DAC', 'TAC', 'TCC', 'TACC']
	selected_repdna = [name for name in repdna_names if name in selected]

	for preprocessed_fasta, label_file in preprocessed_fastas:
		command_map = {
			'NAC': [sys.executable, 'MathFeature/methods/ExtractionTechniques.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'NAC.csv'), '-l', label_file, '-t', 'NAC', '-seq', '1'],
			'DNC': [sys.executable, 'MathFeature/methods/ExtractionTechniques.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'DNC.csv'), '-l', label_file, '-t', 'DNC', '-seq', '1'],
			'TNC': [sys.executable, 'MathFeature/methods/ExtractionTechniques.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'TNC.csv'), '-l', label_file, '-t', 'TNC', '-seq', '1'],
			'kGap_di': [sys.executable, 'MathFeature/methods/Kgap.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'kGap_di.csv'), '-l', label_file, '-k', '1', '-bef', '1', '-aft', '2', '-seq', '1'],
			'kGap_tri': [sys.executable, 'MathFeature/methods/Kgap.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'kGap_tri.csv'), '-l', label_file, '-k', '1', '-bef', '1', '-aft', '3', '-seq', '1'],
			'ORF': [sys.executable, 'MathFeature/methods/CodingClass.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'ORF.csv'), '-l', label_file],
			'Fickett': [sys.executable, 'other-methods/mathfeature-modified/methods/FickettScore.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'Fickett.csv'), '-l', label_file, '-seq', '1'],
			'Shannon': [sys.executable, 'MathFeature/methods/EntropyClass.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'Shannon.csv'), '-l', label_file, '-k', '5', '-e', 'Shannon'],
			'FourierBinary': [sys.executable, 'other-methods/mathfeature-modified/methods/FourierClass.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'FourierBinary.csv'), '-l', label_file, '-r', '1'],
			'FourierComplex': [sys.executable, 'other-methods/FourierClass.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'FourierComplex.csv'), '-l', label_file, '-r', '6'],
			'Tsallis': [sys.executable, 'other-methods/TsallisEntropy.py', '-i', preprocessed_fasta,
				'-o', os.path.join(selected_path, 'Tsallis.csv'), '-l', label_file, '-k', '5', '-q', '2.3'],
		}
		commands = []
		for descriptor, command in command_map.items():
			if descriptor in selected:
				commands.append((descriptor, command))
				dataset = os.path.join(selected_path, f'{descriptor}.csv')
				if dataset not in datasets:
					datasets.append(dataset)

		if selected_repdna:
			dataset = os.path.join(selected_path, 'repDNA.csv')
			commands.append(('repDNA', [
				sys.executable, 'other-methods/repDNA/repDNA-feat.py', '--file', preprocessed_fasta,
				'--output', dataset, '--label', label_file, '--descriptors', *selected_repdna
			]))
			if dataset not in datasets:
				datasets.append(dataset)

		run_feature_commands(commands, os.path.join(selected_path, 'logs'), n_cpu)

	return save_selected_test(datasets, column_train, foutput)

##########################################################################
##########################################################################

if __name__ == '__main__':
	print(r'''
####################################################################################################
####################################################################################################
##  ____   _                        _          __  __  _           ______         _____  _______  ##
## |  _ \ (_)          /\          | |        |  \/  || |         |  ____|/\     / ____||__   __| ##
## | |_) | _   ___    /  \   _   _ | |_  ___  | \  / || |  ______ | |__  /  \   | (___     | |    ##
## |  _ < | | / _ \  / /\ \ | | | || __|/ _ \ | |\/| || | |______||  __|/ /\ \   \___ \    | |    ##
## | |_) || || (_) |/ ____ \| |_| || |_| (_) || |  | || |____     | |  / ____ \  ____) |   | |    ##
## |____/ |_| \___//_/    \_\\__,_| \__|\___/ |_|  |_||______|    |_| /_/    \_\|_____/    |_|    ##
##                                                                                                ##
##          Empowering Breakthroughs in Life Sciences with End-to-End Machine Learning            ##
##                                                                                                ##
##                                   Engineering module                                           ##
##                                                                                                ##
####################################################################################################
####################################################################################################
	''')
	parser = argparse.ArgumentParser()
	parser.add_argument('-fasta_train', '--fasta_train', nargs='+',
						help='fasta format file, e.g., fasta/ncRNA.fasta'
							'fasta/lncRNA.fasta fasta/circRNA.fasta')
	parser.add_argument('-fasta_label_train', '--fasta_label_train', nargs='+',
						help='labels for fasta files, e.g., ncRNA lncRNA circRNA')
	parser.add_argument('-fasta_test', '--fasta_test', nargs='+',
						help='fasta format file, e.g., fasta/ncRNA fasta/lncRNA fasta/circRNA')
	parser.add_argument('-fasta_label_test', '--fasta_label_test', nargs='+',
						help='labels for fasta files, e.g., ncRNA lncRNA circRNA')
	parser.add_argument('-dtype', '--dtype', default="DNA/RNA", help='Data type - DNA/RNA, Protein, Structured')
	parser.add_argument('-task', '--task', default=0, help='Machine learning task - 0: Classification, 1: Regression - Default: Classification')
	parser.add_argument('-estimations', '--estimations', default=200, help='number of estimations - BioAutoML-FAST - default = 200')
	parser.add_argument('-patience', '--patience', default=80, help='number of trials before early stopping - default = 80')
	parser.add_argument('-tuning', '--tuning', default=150, help='number of trials for hyperparameter tuning - default = 150')
	parser.add_argument('-difference', '--difference', default=0.001, help='difference before early stopping - default = 0.001')
	parser.add_argument('-n_cpu', '--n_cpu', default=-1, help='number of cpus - default = all')
	parser.add_argument('-search_jobs', '--search_jobs', default=1, help='parallel Optuna workers; default 1 for repeatable trial ordering')
	parser.add_argument('--stage2_gate', action='store_true', help='Opt in to the training-CV default LightGBM fallback')
	parser.add_argument('--homology_aware', action='store_true', help='Automatically group sequence CV folds using MMseqs2 and audit train-test similarity')
	add_homology_arguments(parser)
	parser.add_argument('--stage2_gate_margin_sd', type=float, default=0.5, help='Nonnegative default-model fold SD multiplier (default: 0.5)')
	parser.add_argument('-seed', '--seed', default=63, help='random seed for cross-validation and learners - default = 63')
	parser.add_argument('-search_seed', '--search_seed', default=None, help='Optuna sampler seed; defaults to --seed')
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')

	args = parser.parse_args()
	homology_identity, homology_coverage = resolve_homology_arguments(parser, args)
	if not np.isfinite(args.stage2_gate_margin_sd) or args.stage2_gate_margin_sd < 0:
		parser.error('--stage2_gate_margin_sd must be finite and nonnegative')
	try:
		fasta_train, fasta_label_train = prepare_fasta_inputs(args.fasta_train, args.fasta_label_train)
	except ValueError as error:
		parser.error(str(error))
	fasta_test = args.fasta_test
	fasta_label_test = args.fasta_label_test
	if fasta_test or fasta_label_test:
		try:
			fasta_test, fasta_label_test = prepare_fasta_inputs(fasta_test, fasta_label_test)
		except ValueError as error:
			parser.error(str(error))
	dtype = args.dtype
	task = int(args.task)
	estimations = int(args.estimations)
	patience = int(args.patience)
	tuning = int(args.tuning)
	difference = float(args.difference)
	n_cpu = int(args.n_cpu)
	search_jobs = int(args.search_jobs) if args.search_jobs is not None else None
	random_seed = int(args.seed)
	optimization_seed = int(args.search_seed) if args.search_seed is not None else random_seed
	random.seed(random_seed)
	np.random.seed(random_seed)
	foutput = str(args.output)

	for fasta in fasta_train:
		if os.path.exists(fasta) is True:
			print('Train - %s: Found File' % fasta)
		else:
			print('Train - %s: File not exists' % fasta)
			sys.exit()

	if fasta_test:
		for fasta in fasta_test:
			if os.path.exists(fasta) is True:
				print('Test - %s: Found File' % fasta)
			else:
				print('Test - %s: File not exists' % fasta)
				sys.exit()

	start_time = time.time()
	homology_report = prepare_homology(
		fasta_train, fasta_label_train, fasta_test, fasta_label_test, dtype, task,
		random_seed, os.path.join(foutput, 'homology'), args.homology_aware, n_cpu, homology_identity, homology_coverage)

	folder_name = foutput.split("/")[-1]

	if folder_name == "run_1" or "run" not in folder_name:
		if dtype == "protein" or dtype == "Protein":
			fnameseqtrain, fnameseqtest, ftrain, ftrain_labels, \
				ftest, ftest_labels = feature_extraction_aminoacid(fasta_train, fasta_label_train,
															None, None, foutput, n_cpu)
		elif dtype == "dnarna" or dtype == "DNA/RNA":
			fnameseqtrain, fnameseqtest, ftrain, ftrain_labels, \
				ftest, ftest_labels = feature_extraction_nucleotide(fasta_train, fasta_label_train,
															None, None, foutput, n_cpu)
	else:
		dataset = "/".join(foutput.split("/")[:-1])
		dataset_run1 = os.path.join(dataset, "run_1")

		if os.path.exists(dataset_run1):
			dataset_run1_feat = os.path.join(dataset_run1, "feat_extraction")

			fnameseqtrain, ftrain, ftrain_labels = os.path.join(dataset_run1_feat, "fnameseqtrain.csv"), os.path.join(dataset_run1_feat, "ftrain.csv"), os.path.join(dataset_run1_feat, "flabeltrain.csv")

			fnameseqtest, ftest, ftest_labels = '', '', ''

	if dtype == "protein" or dtype == "Protein":
		path_train, path_test, train_best, test_best = \
			feature_engineering_aminoacid(task, estimations, fnameseqtrain, ftrain, ftrain_labels, '', foutput, n_cpu, search_jobs, homology_report)
	elif dtype == "dnarna" or dtype == "DNA/RNA":
		path_train, path_test, train_best, test_best = \
			feature_engineering_nucleotide(task, estimations, fnameseqtrain, ftrain, ftrain_labels, '', foutput, n_cpu, search_jobs, homology_report)

	if fasta_test:
		selected_descriptors = os.path.join(foutput, 'best_descriptors', 'selected_descriptors.csv')
		if dtype == "protein" or dtype == "Protein":
			fnameseqtest, path_test, ftest_labels, test_best = feature_extraction_aminoacid_test(
				fasta_test, fasta_label_test, selected_descriptors, train_best.columns, foutput, n_cpu
			)
		elif dtype == "dnarna" or dtype == "DNA/RNA":
			fnameseqtest, path_test, ftest_labels, test_best = feature_extraction_nucleotide_test(
				fasta_test, fasta_label_test, selected_descriptors, train_best.columns, foutput, n_cpu
			)

	cost = (time.time() - start_time) / 60
	print('Computation time - Pipeline - Automated Feature Engineering: %s minutes' % cost)

	subprocess.run([sys.executable, 'generation.py', '-task', str(task), '-tuning', str(tuning), '-train', path_train,
					'-train_label', ftrain_labels, '-test', path_test, 
					'-test_label', ftest_labels, '-train_nameseq', fnameseqtrain,
					'-test_nameseq', fnameseqtest, '-n_cpu', str(n_cpu),
					'-seed', str(random_seed), '-search_seed', str(optimization_seed),
					'-output', foutput]
						+ (['-search_jobs', str(search_jobs)] if search_jobs is not None else [])
						+ ['--stage2_gate_margin_sd', str(args.stage2_gate_margin_sd)]
						+ (['--stage2_gate'] if args.stage2_gate else [])
						+ (['--homology_aware', '--homology_identity', str(homology_identity),
						    '--homology_coverage', str(homology_coverage)] if args.homology_aware else []), check=True)

##########################################################################
##########################################################################
