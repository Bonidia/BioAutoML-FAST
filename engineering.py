import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import polars as pl
import argparse
import subprocess
import shutil
import sys
import os.path
import time
import xgboost as xgb
import lightgbm as lgb
import optuna
import pygad
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, roc_auc_score, matthews_corrcoef, average_precision_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from subprocess import Popen
from multiprocessing import Manager
import numpy as np
from Bio import SeqIO

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

def objective_nucleotide(trial, train, task, y):
	"""Automated Feature Engineering - Optuna - Objective Function - Bayesian Optimization"""

	# Define search space
	space = {
		'NAC': trial.suggest_categorical('NAC', [0, 1]),
		'DNC': trial.suggest_categorical('DNC', [0, 1]),
		'TNC': trial.suggest_categorical('TNC', [0, 1]),
		'kGap_di': trial.suggest_categorical('kGap_di', [0, 1]),
		'kGap_tri': trial.suggest_categorical('kGap_tri', [0, 1]),
		'ORF': trial.suggest_categorical('ORF', [0, 1]),
		'Fickett': trial.suggest_categorical('Fickett', [0, 1]),
		'Shannon': trial.suggest_categorical('Shannon', [0, 1]),
		'FourierBinary': trial.suggest_categorical('FourierBinary', [0, 1]),
		'FourierComplex': trial.suggest_categorical('FourierComplex', [0, 1]),
		'Tsallis': trial.suggest_categorical('Tsallis', [0, 1]),
		'Revkmer': trial.suggest_categorical('Revkmer', [0, 1]),
		'PseDNC': trial.suggest_categorical('PseDNC', [0, 1]),
		'PseKNC': trial.suggest_categorical('PseKNC', [0, 1]),
		'SC-PseDNC': trial.suggest_categorical('SC-PseDNC', [0, 1]),
		'SC-PseTNC': trial.suggest_categorical('SC-PseTNC', [0, 1]),
		'DAC': trial.suggest_categorical('DAC', [0, 1]),
		'TAC': trial.suggest_categorical('TAC', [0, 1]),
		'TCC': trial.suggest_categorical('TCC', [0, 1]),
		'TACC': trial.suggest_categorical('TACC', [0, 1])
	}

	# Descriptor indices
	descriptors = {
		'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
		'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
		'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
		'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
		'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
		'Tsallis': list(range(459, 464)), 'Revkmer': list(range(464, 508)),
		'PseDNC': list(range(508, 527)), 'PseKNC': list(range(527, 592)),
		'SC-PseDNC': list(range(592, 646)), 'SC-PseTNC': list(range(646, 734)),
		'DAC': list(range(734, 810)), 'TAC': list(range(810, 834)),
		'TCC': list(range(834, 1098)), 'TACC': list(range(1098, 1386))
	}

	index = []
	for d, inds in descriptors.items():
		if space[d] == 1:
			index.extend(inds)

	if len(index) == 0:
		raise optuna.TrialPruned()

	# === Task Handling ===
	if task == 0:
		model = Pipeline([
			("imputer", SimpleImputer(strategy="mean")),
			("clf", lgb.LGBMClassifier(
				random_state=63,
				verbosity=-1
			))
		])
		cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)
	elif task == 1:
		model = Pipeline([
			("imputer", SimpleImputer(strategy="mean")),
			("reg", lgb.LGBMRegressor(
				random_state=63,
				verbosity=-1
			))
		])
		cv = KFold(n_splits=5, shuffle=True, random_state=63)
	else:
		raise ValueError("Invalid task. Use 0 (classification) or 1 (regression).")

	if isinstance(y, list):
		y = np.array(y)

	# === Cross-Validation ===
	fold_scores = []
	X_subset = train.iloc[:, index]

	try:
		# Manual CV loop to enable pruning
		for step, (train_idx, val_idx) in enumerate(cv.split(X_subset, y)):
			X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
			y_train, y_val = y[train_idx], y[val_idx]
			
			model.fit(X_train, y_train)
			preds = model.predict(X_val)
			
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
	except Exception:
		rint("Trial failed with exception:")
		print(type(e).__name__, str(e))
		raise optuna.TrialPruned()

	return metric

def feature_engineering_nucleotide(task, estimations, fnameseqtrain, train, train_labels, test, foutput):
	"""Select the best subset of nucleotide descriptors via Bayesian optimization (Optuna TPE).

	Treats each descriptor group as a binary on/off variable and maximises MCC (task=0)
	or minimises RMSE (task=1) using 5-fold CV with LightGBM.
	Saves selected_descriptors.csv, best_train.csv, and best_test.csv under foutput/best_descriptors/.
	Returns (path_btrain, path_btest, btrain_df, btest_df).
	"""
	print('Automated Feature Engineering - Bayesian Optimization')

	df_x = pd.read_csv(train)
	mgr = Manager()
	ns = mgr.Namespace()
	ns.df = df_x
	
	path_bio = foutput + '/best_descriptors'
	if not os.path.exists(path_bio):
		os.mkdir(path_bio)

	param = {'NAC': [0, 1], 'DNC': [0, 1],
			'TNC': [0, 1], 'kGap_di': [0, 1], 'kGap_tri': [0, 1],
			'ORF': [0, 1], 'Fickett': [0, 1],
			'Shannon': [0, 1], 
			'FourierBinary': [0, 1],
			'FourierComplex': [0, 1], 
			'Tsallis': [0, 1],
			'Revkmer': [0, 1], 'PseDNC': [0, 1],
			'PseKNC': [0, 1], 'SC-PseDNC': [0, 1],
			'SC-PseTNC': [0, 1], 
			'DAC': [0, 1],
			'TAC': [0, 1], 'TCC': [0, 1],
			'TACC': [0, 1]}
	
	if task == 0:
		labels = pd.read_csv(train_labels)
		le = LabelEncoder()
		y = le.fit_transform(labels)
		direction = "maximize"
	elif task == 1:
		y = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(fnameseqtrain)["nameseq"].to_list()]
		direction = "minimize"

	func = lambda trial: objective_nucleotide(trial, ns.df, task, y)
	
	early_stopping = EarlyStoppingCallback(
		patience=patience,
		min_delta=difference
	)

	results = optuna.create_study(
		direction=direction,
		sampler=optuna.samplers.TPESampler(n_startup_trials=30, multivariate=True, group=True, constant_liar=True)
	)

	results.optimize(
		func,
		n_trials=estimations,
		timeout=10_800,
		show_progress_bar=True,
		callbacks=[early_stopping],
		n_jobs=16
	)

	best_tuning = results.best_params
	print(best_tuning)
	
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				'Tsallis': list(range(459, 464)), 'Revkmer': list(range(464, 508)),
					'PseDNC': list(range(508, 527)), 'PseKNC': list(range(527, 592)),
					'SC-PseDNC': list(range(592, 646)), 'SC-PseTNC': list(range(646, 734)),
					'DAC': list(range(734, 810)), 'TAC': list(range(810, 834)),
					'TCC': list(range(834, 1098)), 'TACC': list(range(1098, 1386))
				}

	# Get indices of selected descriptors
	index = []
	descriptor_presence = {}
	for descriptor, ind in descriptors.items():
		result = best_tuning[descriptor]
		if result == 1:
			index.extend(ind)
			descriptor_presence[descriptor] = 1
		else:
			descriptor_presence[descriptor] = 0

	# Save presence/absence table
	df_presence = pd.DataFrame([descriptor_presence])
	df_presence.to_csv(os.path.join(path_bio, 'selected_descriptors.csv'), index=False)
	
	if test != '':
		df_test = pd.read_csv(test)

	btrain = ns.df.iloc[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.to_csv(path_btrain, index=False)

	if test != '':
		btest = df_test.iloc[:, index]
		path_btest = path_bio + '/best_test.csv'
		btest.to_csv(path_btest, index=False)
	else:
		btest, path_btest = '', ''

	return path_btrain, path_btest, btrain, btest

def objective_aminoacid(trial, train, task, y):

	"""Automated Feature Engineering - Optuna - Objective Function - Bayesian Optimization"""

	space = {
			'Shannon': trial.suggest_categorical('Shannon', [0, 1]),
			'Tsallis_23': trial.suggest_categorical('Tsallis_23', [0, 1]),
			'Tsallis_30': trial.suggest_categorical('Tsallis_30', [0, 1]),
			'Tsallis_40': trial.suggest_categorical('Tsallis_40', [0, 1]),
			'ComplexNetworks': trial.suggest_categorical('ComplexNetworks', [0, 1]),
			'kGap': trial.suggest_categorical('kGap', [0, 1]),
			'AAC': trial.suggest_categorical('AAC', [0, 1]),
			'DPC': trial.suggest_categorical('DPC', [0, 1]),
			'CKSAAP': trial.suggest_categorical('CKSAAP', [0, 1]),
			'DDE': trial.suggest_categorical('DDE', [0, 1]),
			'GAAC': trial.suggest_categorical('GAAC', [0, 1]),
			'CKSAAGP': trial.suggest_categorical('CKSAAGP', [0, 1]),
			'GDPC': trial.suggest_categorical('GDPC', [0, 1]),
			'GTPC': trial.suggest_categorical('GTPC', [0, 1]),
			'CTDC': trial.suggest_categorical('CTDC', [0, 1]),
			'CTDT': trial.suggest_categorical('CTDT', [0, 1]),
			'CTDD': trial.suggest_categorical('CTDD', [0, 1]),
			'CTriad': trial.suggest_categorical('CTriad', [0, 1]),
			'KSCTriad': trial.suggest_categorical('KSCTriad', [0, 1]),
			'Global': trial.suggest_categorical('Global', [0, 1]),
			'Peptide': trial.suggest_categorical('Peptide', [0, 1]),
			'Fourier_Integer': trial.suggest_categorical('Fourier_Integer', [0, 1]),
			'Fourier_EIIP': trial.suggest_categorical('Fourier_EIIP', [0, 1])
	}

	descriptors = {'Shannon': list(range(0, 5)), 'Tsallis_23': list(range(5, 10)),
				'Tsallis_30': list(range(10, 15)), 'Tsallis_40': list(range(15, 20)),
				'ComplexNetworks': list(range(20, 98)), 'kGap': list(range(98, 498)),
				'AAC': list(range(498, 518)),
				'DPC': list(range(518, 918)),
				'CKSAAP': list(range(918, 3318)), 
				'DDE': list(range(3318, 3718)),
				'GAAC': list(range(3718, 3723)),
				'CKSAAGP': list(range(3723, 3873)),
				'GDPC': list(range(3873, 3898)),
				'GTPC': list(range(3898, 4023)),
				'CTDC': list(range(4023, 4062)),
				'CTDT': list(range(4062, 4101)),
				'CTDD': list(range(4101, 4296)),
				'CTriad': list(range(4296, 4639)),
				'KSCTriad': list(range(4639, 4982)), 
				'Global': list(range(4982, 4992)),
				'Peptide': list(range(4992, 5008)),
				'Fourier_Integer': list(range(5008, 5027)),
				'Fourier_EIIP': list(range(5027, 5046))
	}

	index = []
	for d, inds in descriptors.items():
		if space[d] == 1:
			index.extend(inds)

	if len(index) == 0:
		raise optuna.TrialPruned()

	# === Task Handling ===
	if task == 0:
		model = Pipeline([
			("imputer", SimpleImputer(strategy="mean")),
			("clf", lgb.LGBMClassifier(
				random_state=63,
				verbosity=-1
			))
		])
		cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)
	elif task == 1:
		model = Pipeline([
			("imputer", SimpleImputer(strategy="mean")),
			("reg", lgb.LGBMRegressor(
				random_state=63,
				verbosity=-1
			))
		])
		cv = KFold(n_splits=5, shuffle=True, random_state=63)
	else:
		raise ValueError("Invalid task. Use 0 (classification) or 1 (regression).")

	if isinstance(y, list):
		y = np.array(y)
		
	fold_scores = []
	X_subset = train.iloc[:, index]

	try:
		for step, (train_idx, val_idx) in enumerate(cv.split(X_subset, y)):
			X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
			y_train, y_val = y[train_idx], y[val_idx]

			model.fit(X_train, y_train)
			preds = model.predict(X_val)

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

def feature_engineering_aminoacid(task, estimations, fnameseqtrain, train, train_labels, test, foutput):
	"""Select the best subset of amino acid descriptors via Bayesian optimization (Optuna TPE).

	Treats each descriptor group as a binary on/off variable and maximises MCC (task=0)
	or minimises RMSE (task=1) using 5-fold CV with LightGBM.
	Saves selected_descriptors.csv, best_train.csv, and best_test.csv under foutput/best_descriptors/.
	Returns (path_btrain, path_btest, btrain_df, btest_df).
	"""

	print('Automated Feature Engineering - Bayesian Optimization')

	df_x = pd.read_csv(train)
	mgr = Manager()
	ns = mgr.Namespace()
	ns.df = df_x

	path_bio = foutput + '/best_descriptors'
	if not os.path.exists(path_bio):
		os.mkdir(path_bio)

	param = {'Shannon': [0, 1], 'Tsallis_23': [0, 1],
			'Tsallis_30': [0, 1], 'Tsallis_40': [0, 1],
			'ComplexNetworks': [0, 1],
			'kGap': [0, 1],
			'AAC': [0, 1], 'DPC': [0, 1],
			'CKSAAP': [0, 1],
			'DDE': [0, 1],
			'GAAC': [0, 1],
			'CKSAAGP': [0, 1],
			'GDPC': [0, 1],
			'GTPC': [0, 1],
			'CTDC': [0, 1],
			'CTDT': [0, 1],
			'CTDD': [0, 1],
			'CTriad': [0, 1],
			'KSCTriad': [0, 1],
			'Global': [0, 1],
			'Peptide': [0, 1],
			'Fourier_Integer': [0, 1],
			'Fourier_EIIP': [0, 1]}

	if task == 0:
		labels = pd.read_csv(train_labels)
		le = LabelEncoder()
		y = le.fit_transform(labels)
		direction = "maximize"
	elif task == 1:
		y = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(fnameseqtrain)["nameseq"].to_list()]
		direction = "minimize"

	func = lambda trial: objective_aminoacid(trial, ns.df, task, y)

	early_stopping = EarlyStoppingCallback(
		patience=patience,
		min_delta=difference
	)

	results = optuna.create_study(
		direction=direction,
		sampler=optuna.samplers.TPESampler(n_startup_trials=30, multivariate=True, group=True, constant_liar=True)
	)

	results.optimize(
		func,
		n_trials=estimations,
		timeout=7200,
		show_progress_bar=True,
		callbacks=[early_stopping],
		n_jobs=16
	)

	best_tuning = results.best_params
	print(best_tuning)
	
	descriptors = {'Shannon': list(range(0, 5)), 'Tsallis_23': list(range(5, 10)),
				'Tsallis_30': list(range(10, 15)), 'Tsallis_40': list(range(15, 20)),
				'ComplexNetworks': list(range(20, 98)), 'kGap': list(range(98, 498)),
				'AAC': list(range(498, 518)),
				'DPC': list(range(518, 918)),
				'CKSAAP': list(range(918, 3318)), 
				'DDE': list(range(3318, 3718)),
				'GAAC': list(range(3718, 3723)),
				'CKSAAGP': list(range(3723, 3873)),
				'GDPC': list(range(3873, 3898)),
				'GTPC': list(range(3898, 4023)),
				'CTDC': list(range(4023, 4062)),
				'CTDT': list(range(4062, 4101)),
				'CTDD': list(range(4101, 4296)),
				'CTriad': list(range(4296, 4639)),
				'KSCTriad': list(range(4639, 4982)), 
				'Global': list(range(4982, 4992)),
				'Peptide': list(range(4992, 5008)),
				'Fourier_Integer': list(range(5008, 5027)),
				'Fourier_EIIP': list(range(5027, 5046)),}

	index = []
	# Determine which descriptors were selected
	descriptor_presence = {}
	for descriptor, ind in descriptors.items():
		result = best_tuning[descriptor]
		if result == 1:
			index.extend(ind)
			descriptor_presence[descriptor] = 1
		else:
			descriptor_presence[descriptor] = 0

	# Save presence/absence summary CSV
	df_presence = pd.DataFrame([descriptor_presence])
	df_presence.to_csv(os.path.join(path_bio, 'selected_descriptors.csv'), index=False)

	if test != '':
		df_test = pd.read_csv(test)

	btrain = ns.df.iloc[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.to_csv(path_btrain, index=False)

	if test != '':
		btest = df_test.iloc[:, index]
		path_btest = path_bio + '/best_test.csv'
		btest.to_csv(path_btest, index=False)
	else:
		btest, path_btest = '', ''

	return path_btrain, path_btest, btrain, btest

def feature_extraction_aminoacid(ftrain, ftrain_labels, ftest, ftest_labels, foutput):
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

	print('Extracting features...')

	for fasta_files, label_files, split_type in input_groups:
		for fasta_file, label_file in zip(fasta_files, label_files):
			# Preprocess file
			file_name = os.path.basename(fasta_file)
			preprocessed_fasta = os.path.join(path, split_type, f'pre_{file_name}')
			
			subprocess.run([
				'python', 'other-methods/preprocessing.py',
				'-i', fasta_file,
				'-o', preprocessed_fasta,
				'-s', split_type,
				'-d', "Protein",
			], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			
			if split_type == 'train':
				with open(preprocessed_fasta) as handle:
					sequence_train.update(str(record.id) for record in SeqIO.parse(handle, "fasta"))
			
			fasta_list.append(preprocessed_fasta)
			
			# Define all feature extraction commands
			commands = [
				['python', 'other-methods/EntropyClass.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'Shannon.csv'),
				'-l', label_file, '-k', '5', '-e', 'Shannon'],

				['python', 'other-methods/TsallisEntropy.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'Tsallis_23.csv'),
				'-l', label_file, '-k', '5', '-q', '2.3'],
				
				['python', 'other-methods/TsallisEntropy.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'Tsallis_30.csv'),
				'-l', label_file, '-k', '5', '-q', '3.0'],
				
				['python', 'other-methods/TsallisEntropy.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'Tsallis_40.csv'),
				'-l', label_file, '-k', '5', '-q', '4.0'],
				
				['python', 'MathFeature/methods/ComplexNetworksClass-v2.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'ComplexNetworks.csv'),
				'-l', label_file, '-k', '3'],
				
				['python', 'MathFeature/methods/Kgap.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'kGap.csv'),
				'-l', label_file, '-k', '1', '-bef', '1', '-aft', '1', '-seq', '3'],
				
				['python', 'other-methods/ExtractionTechniques-Protein.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'AAC.csv'),
				'-l', label_file, '-t', 'AAC'],
				
				['python', 'other-methods/ExtractionTechniques-Protein.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'DPC.csv'),
				'-l', label_file, '-t', 'DPC'],
				
				['python', 'other-methods/iFeature-modified/iFeature.py',
				'--file', preprocessed_fasta, '--type', 'All',
				'--label', label_file, '--out', os.path.join(path, 'iFeature-features.csv')],
				
				['python', 'other-methods/modlAMP-modified/descriptors.py',
				'-option', 'global', '-label', label_file,
				'-input', preprocessed_fasta, '-output', os.path.join(path, 'Global.csv')],
				
				['python', 'other-methods/modlAMP-modified/descriptors.py',
				'-option', 'peptide', '-label', label_file,
				'-input', preprocessed_fasta, '-output', os.path.join(path, 'Peptide.csv')]
			]
			
			log_dir = os.path.join(path, 'logs')
			os.makedirs(log_dir, exist_ok=True)  # make sure the folder exists

			processes = []
			for cmd, dataset in zip(commands, datasets):
				log_path = os.path.join(log_dir, f"{dataset.split('/')[-1].split('.csv')[0]}.log")
				with open(log_path, "a") as log_file:
					p = subprocess.Popen(
						cmd,
						stdout=log_file,
						stderr=subprocess.STDOUT
					)
					processes.append(p)

			# wait for all to finish
			for p in processes:
				p.wait()

	# Process Fourier features
	labels_list = ftrain_labels + (ftest_labels if ftest else [])
	text_input = '\n'.join(f'{fasta}\n{label}' for fasta, label in zip(fasta_list, labels_list))

	fourier_datasets = [
		('Fourier_Integer.csv', '6', 'Integer_Fourier_'),
		('Fourier_EIIP.csv', '8', 'EIIP_Fourier_')
	]

	for fname, r_val, prefix in fourier_datasets:
		dataset = os.path.join(path, fname)
		subprocess.run([
			'python', 'MathFeature/methods/Mappings-Protein.py',
			'-n', str(len(fasta_list)), '-o', dataset, '-r', r_val
		], text=True, input=text_input, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		
		with open(dataset) as f:
			max_cols = max(len(line.split(",")) for line in f)
		
		colnames = [f'{prefix}{i}' for i in range(max_cols)]
		df = pd.read_csv(dataset, names=colnames, header=0)
		df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
		df.to_csv(dataset, index=False)
		datasets.append(dataset)

	"""Concatenating all the extracted features"""
	
	if datasets:
		dfs_list = [
			pl.read_csv(f, infer_schema=False)
			.select(pl.all().exclude("nameseq"), pl.col("nameseq"))
			.filter(~pl.col("nameseq").str.contains("nameseq")) 
			.set_sorted("nameseq")
			for f in datasets
		]

		dataframes = pl.concat(dfs_list, how="align")

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
	X_train.select(pl.all().exclude(["split_type", "nameseq", "label"])).write_csv(ftrain)
	
	fnameseqtest, ftest, flabeltest = '', '', ''

	if fasta_test:
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

def feature_extraction_nucleotide(ftrain, ftrain_labels, ftest, ftest_labels, foutput):
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

	print('Extracting features...')

	for fasta_files, label_files, split_type in input_groups:
		for fasta_file, label_file in zip(fasta_files, label_files):
			# Preprocess file
			file_name = os.path.basename(fasta_file)
			preprocessed_fasta = os.path.join(path, split_type, f'pre_{file_name}')
			
			subprocess.run([
				'python', 'other-methods/preprocessing.py',
				'-i', fasta_file,
				'-o', preprocessed_fasta,
				'-s', split_type,
				'-d', "DNA/RNA",
			], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			
			if split_type == 'train':
				with open(preprocessed_fasta) as handle:
					sequence_train.update(str(record.id) for record in SeqIO.parse(handle, "fasta"))
			
			fasta_list.append(preprocessed_fasta)
			
			# Define all feature extraction commands
			commands = [
				['python', 'MathFeature/methods/ExtractionTechniques.py',
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'NAC.csv'), '-l', label_file,
				'-t', 'NAC', '-seq', '1'],

				['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'DNC.csv'), '-l', label_file,
				'-t', 'DNC', '-seq', '1'],

				['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'TNC.csv'), '-l', label_file,
				'-t', 'TNC', '-seq', '1'],

				['python', 'MathFeature/methods/Kgap.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'kGap_di.csv'), '-l',
				label_file, '-k', '1', '-bef', '1', '-aft', '2', '-seq', '1'],

				['python', 'MathFeature/methods/Kgap.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'kGap_tri.csv'), '-l',
				label_file, '-k', '1', '-bef', '1', '-aft', '3', '-seq', '1'],

				['python', 'MathFeature/methods/CodingClass.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'ORF.csv'), '-l', label_file],

				['python', 'MathFeature/methods/FickettScore.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'Fickett.csv'), '-l', label_file,
				'-seq', '1'],

				['python', 'MathFeature/methods/EntropyClass.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'Shannon.csv'), '-l', label_file,
				'-k', '5', '-e', 'Shannon'],

				['python', 'MathFeature/methods/FourierClass.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'FourierBinary.csv'), '-l', label_file,
				'-r', '1'],

				['python', 'other-methods/FourierClass.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'FourierComplex.csv'), '-l', label_file,
				'-r', '6'],

				['python', 'other-methods/TsallisEntropy.py', '-i',
				preprocessed_fasta, '-o', os.path.join(path, 'Tsallis.csv'), '-l', label_file,
				'-k', '5', '-q', '2.3'],

				['python', 'other-methods/repDNA/repDNA-feat.py', '--file',
				preprocessed_fasta, '--output', os.path.join(path, 'repDNA.csv'), '--label', label_file]
			]
			
			log_dir = os.path.join(path, 'logs')
			os.makedirs(log_dir, exist_ok=True)  # make sure the folder exists

			processes = []
			for cmd, dataset in zip(commands, datasets):
				log_path = os.path.join(log_dir, f"{dataset.split('/')[-1].split('.csv')[0]}.log")
				with open(log_path, "w") as log_file:
					p = subprocess.Popen(
						cmd,
						stdout=log_file,
						stderr=subprocess.STDOUT
					)
					processes.append(p)

			# wait for all to finish
			for p in processes:
				p.wait()

	"""Concatenating all the extracted features"""
	
	if datasets:
		dfs_list = [
			pl.read_csv(f, infer_schema=False)
			.select(pl.all().exclude("nameseq"), pl.col("nameseq"))
			.filter(~pl.col("nameseq").str.contains("nameseq")) 
			.set_sorted("nameseq")
			for f in datasets
		]

		dataframes = pl.concat(dfs_list, how="align")
		
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
	X_train.select(pl.all().exclude(["split_type", "nameseq", "label"])).write_csv(ftrain)
	
	fnameseqtest, ftest, flabeltest = '', '', ''

	if fasta_test:
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
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')

	args = parser.parse_args()
	fasta_train = args.fasta_train
	fasta_label_train = args.fasta_label_train
	fasta_test = args.fasta_test
	fasta_label_test = args.fasta_label_test
	dtype = args.dtype
	task = int(args.task)
	estimations = int(args.estimations)
	patience = int(args.patience)
	tuning = int(args.tuning)
	difference = float(args.difference)
	n_cpu = int(args.n_cpu)
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

	folder_name = foutput.split("/")[-1]

	if folder_name == "run_1" or "run" not in folder_name:
		if dtype == "protein" or dtype == "Protein":
			fnameseqtrain, fnameseqtest, ftrain, ftrain_labels, \
				ftest, ftest_labels = feature_extraction_aminoacid(fasta_train, fasta_label_train,
																	fasta_test, fasta_label_test, foutput)
		elif dtype == "dnarna" or dtype == "DNA/RNA":
			fnameseqtrain, fnameseqtest, ftrain, ftrain_labels, \
				ftest, ftest_labels = feature_extraction_nucleotide(fasta_train, fasta_label_train,
																	fasta_test, fasta_label_test, foutput)
	else:
		dataset = "/".join(foutput.split("/")[:-1])
		dataset_run1 = os.path.join(dataset, "run_1")

		if os.path.exists(dataset_run1):
			dataset_run1_feat = os.path.join(dataset_run1, "feat_extraction")

			fnameseqtrain, ftrain, ftrain_labels = os.path.join(dataset_run1_feat, "fnameseqtrain.csv"), os.path.join(dataset_run1_feat, "ftrain.csv"), os.path.join(dataset_run1_feat, "flabeltrain.csv")

			fnameseqtest, ftest, ftest_labels = '', '', ''
			if os.path.exists(os.path.join(dataset_run1_feat, "ftest.csv")):
				fnameseqtest, ftest, ftest_labels = os.path.join(dataset_run1_feat, "fnameseqtest.csv"), os.path.join(dataset_run1_feat, "ftest.csv"), os.path.join(dataset_run1_feat, "flabeltest.csv") 

	if dtype == "protein" or dtype == "Protein":
		path_train, path_test, train_best, test_best = \
			feature_engineering_aminoacid(task, estimations, fnameseqtrain, ftrain, ftrain_labels, ftest, foutput)
	elif dtype == "dnarna" or dtype == "DNA/RNA":
		path_train, path_test, train_best, test_best = \
			feature_engineering_nucleotide(task, estimations, fnameseqtrain, ftrain, ftrain_labels, ftest, foutput)

	cost = (time.time() - start_time) / 60
	print('Computation time - Pipeline - Automated Feature Engineering: %s minutes' % cost)

	subprocess.run(['python', 'generation.py', '-task', str(task), '-tuning', str(tuning), '-train', path_train,
					'-train_label', ftrain_labels, '-test', path_test, 
					'-test_label', ftest_labels, '-train_nameseq', fnameseqtrain,
					'-test_nameseq', fnameseqtest, '-n_cpu', str(n_cpu), '-output', foutput])

##########################################################################
##########################################################################