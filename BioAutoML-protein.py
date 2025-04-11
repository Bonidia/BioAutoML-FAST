import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import polars as pl
import argparse
import subprocess
# import multiprocessing
import shutil
import sys
import os.path
import time
import xgboost as xgb
import lightgbm as lgb
import optuna
import pygad
# from genetic_selection import GeneticSelectionCV
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
# from sklearn_genetic import GAFeatureSelectionCV
# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, SparkTrials, early_stop
from subprocess import Popen
from multiprocessing import Manager
from Bio import SeqIO

def objective_ga_pygad(ga_instance, solution, solution_idx):

	"""Automated Feature Engineering - Objective Function - Genetic Algorithm"""
	
	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464)), 'repDNA': list(range(464, 734))}
 
 
	desc = ['NAC', 'DNC', 'TNC', 'kGap_di', 'kGap_tri', 'ORF', 'Fickett', 'Shannon', 
         	'FourierBinary', 'FourierComplex', 'Tsallis', 'repDNA']
	for gene in range(0, len(solution)):
		if int(solution[gene]) == 1:
			ind = descriptors[desc[int(gene)]]
			index = index + ind
	
 
	if len(fasta_label_train) > 2:
		score = make_scorer(f1_score, average='weighted')
	else:
		score = make_scorer(balanced_accuracy_score)

	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	metric = cross_val_score(model,
							 df_x[:, index],
        					 y,
							 cv=kfold,
							 scoring=score,
							 n_jobs=n_cpu).mean()
	
	return metric


def feature_engineering_pygad(estimations, train, train_labels, test, foutput):

	"""Automated Feature Engineering - Genetic Algorithm"""

	print('Automated Feature Engineering - Genetic Algorithm')
 
	global df_x, y, model

	le = LabelEncoder()
	df_x = pd.read_csv(train)
	y = le.fit_transform(pd.read_csv(train_labels).values.ravel())

	path_bio = foutput + '/best_descriptors'
	if not os.path.exists(path_bio):
		os.mkdir(path_bio)

	classifier = best_algorithms(df_x, y)
	if classifier == 0:
		model = CatBoostClassifier(thread_count=1, nan_mode='Max',
                                   logging_level='Silent', random_state=63)
	elif classifier == 1:
		model = RandomForestClassifier(n_jobs=1, random_state=63)
        
	elif classifier == 2:
		model = lgb.LGBMClassifier(n_jobs=1, random_state=63)
	else:
		model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=1, random_state=63)

	print('Checking the best descriptors...')
	ga_instance = pygad.GA(num_generations=estimations,
                       num_parents_mating=4,
                       fitness_func=objective_ga_pygad,
                       sol_per_pop=20,
                       num_genes=12,
                       gene_type=int,
                       init_range_low=0,
                       init_range_high=2,
                       parent_selection_type="tournament",
                       keep_parents=8,
                       K_tournament=4,
                       crossover_type="two_points",
                       mutation_type="random",
                       suppress_warnings=True,
                       stop_criteria=["saturate_10"],
                       parallel_processing=n_cpu)
	ga_instance.run()
	best = ga_instance.best_solution()[0]
	print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best))
	print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))
 
	
	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464)), 'repDNA': list(range(464, 734))}


	desc = ['NAC', 'DNC', 'TNC', 'kGap_di', 'kGap_tri', 'ORF', 'Fickett', 'Shannon', 
         	'FourierBinary', 'FourierComplex', 'Tsallis', 'repDNA']
 
	for gene in range(0, len(best)-1):
		if int(gene) == 1:
			ind = descriptors[desc[int(gene)]]
			index = index + ind

	if test != '':
		df_test = pd.read_csv(test)

	btrain = df_x.iloc[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.to_csv(path_btrain, index=False)

	if test != '':
		btest = df_test.iloc[:, index]
		path_btest = path_bio + '/best_test.csv'
		btest.to_csv(path_btest, index=False)
	else:
		btest, path_btest = '', ''

	return classifier, path_btrain, path_btest, btrain, btest


def objective(trial, train, train_labels):

	"""Automated Feature Engineering - Optuna - Objective Function - Bayesian Optimization"""
 
	space = {'Shannon': trial.suggest_categorical('Shannon', [0, 1]),
			 'Tsallis_23': trial.suggest_categorical('Tsallis_23', [0, 1]),
			 'Tsallis_30': trial.suggest_categorical('Tsallis_30', [0, 1]),
			 'Tsallis_40': trial.suggest_categorical('Tsallis_40', [0, 1]),
			 'ComplexNetworks': trial.suggest_categorical('ComplexNetworks', [0, 1]),
			 'kGap_di': trial.suggest_categorical('kGap_di', [0, 1]),
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
			 'Fourier_EIIP': trial.suggest_categorical('Fourier_EIIP', [0, 1]),
			#  'EIIP': trial.suggest_categorical('EIIP', [0, 1]),
			#  'AAAF': trial.suggest_categorical('AAAF', [0, 1]),
			 'Classifier': trial.suggest_categorical('Classifier', [1, 2, 3])}

	position = int((len(train.columns) - 5046) / 2)
	index = list()
	descriptors = {'Shannon': list(range(0, 5)), 'Tsallis_23': list(range(5, 10)),
				   'Tsallis_30': list(range(10, 15)), 'Tsallis_40': list(range(15, 20)),
				   'ComplexNetworks': list(range(20, 98)), 'kGap_di': list(range(98, 498)),
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
				#    'EIIP': list(range(5046, (5046 + position))),
				#    'AAAF': list(range((5046 + position), len(train.columns)))} 
 
	for descriptor, ind in descriptors.items():
		if int(space[descriptor]) == 1:
			index = index + ind
	
 
	if int(space['Classifier']) == 0:
		model = CatBoostClassifier(thread_count=1, nan_mode='Max',
								   	   logging_level='Silent', random_state=63)
	elif int(space['Classifier']) == 1:
		model = RandomForestClassifier(n_jobs=1, random_state=63)
	elif int(space['Classifier']) == 2:
		model = lgb.LGBMClassifier(n_jobs=1, random_state=63)
	elif int(space['Classifier']) == 3:
		model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, 
                            			n_jobs=1, random_state=63)


	if len(fasta_label_train) > 2:
		score = make_scorer(f1_score, average='weighted')
	else:
		score = make_scorer(balanced_accuracy_score)

	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	le = LabelEncoder()

	try:
		metric = cross_val_score(model,
								train[:, index],
								le.fit_transform(pl.read_csv(train_labels)),
								cv=kfold,
								scoring=score,
								n_jobs=n_cpu).mean()
	except Exception as e:
		metric = 0.0
		
	return metric


def feature_engineering_optuna(estimations, train, train_labels, test, foutput):

	"""Automated Feature Engineering - Bayesian Optimization"""

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
			 'kGap_di': [0, 1],
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
		  	 'Fourier_EIIP': [0, 1], # 'EIIP': [0, 1],
			#  'AAAF': [0, 1],
			 'Classifier': [1, 2, 3]}
 
	func = lambda trial: objective(trial, ns.df, train_labels)
	
	results = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
	results.optimize(func, n_trials=estimations, timeout=7200, n_jobs=n_cpu, show_progress_bar=True)
 
	best_tuning = results.best_params
 
	print(best_tuning)
	
	position = int((len(df_x.columns) - 5046) / 2)
	index = list()
	descriptors = {'Shannon': list(range(0, 5)), 'Tsallis_23': list(range(5, 10)),
				   'Tsallis_30': list(range(10, 15)), 'Tsallis_40': list(range(15, 20)),
				   'ComplexNetworks': list(range(20, 98)), 'kGap_di': list(range(98, 498)),
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
				#    'EIIP': list(range(5046, (5046 + position))),
				#    'AAAF': list(range((5046 + position), len(df_x.columns)))}
 
	for descriptor, ind in descriptors.items():
		result = param[descriptor][best_tuning[descriptor]]
		if result == 1:
			index = index + ind

	classifier = best_tuning['Classifier']

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

	return classifier, path_btrain, path_btest, btrain, btest


def feature_extraction(ftrain, ftrain_labels, ftest, ftest_labels, foutput):

	"""Extracts the features from the sequences in the fasta files."""

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
		'kGap_di.csv',
		'AAC.csv',
		'DPC.csv',
		'iFeature-features.csv',
		'Global.csv',
		'Peptide.csv'
	]

	datasets = [os.path.join(path, fname) for fname in datasets]

	print('Extracting features with MathFeature...')
 
	for fasta_files, label_files, split_type in input_groups:
		for fasta_file, label_file in zip(fasta_files, label_files):
			# Preprocess file
			file_name = os.path.basename(fasta_file)
			preprocessed_fasta = os.path.join(path, split_type, f'pre_{file_name}')
			
			subprocess.run([
				'python', 'other-methods/preprocessing.py',
				'-i', fasta_file,
				'-o', preprocessed_fasta
			], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			
			if split_type == 'train':
				with open(preprocessed_fasta) as handle:
					sequence_train.update(str(record.id) for record in SeqIO.parse(handle, "fasta"))
			
			fasta_list.append(preprocessed_fasta)
			
			# Define all feature extraction commands
			commands = [
				['python', 'MathFeature/methods/EntropyClass.py',
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
				'-i', preprocessed_fasta, '-o', os.path.join(path, 'kGap_di.csv'),
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
			
			# Run all commands in parallel
			processes = [Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) for cmd in commands]
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

	# dataset = path + '/EIIP.csv'

	# subprocess.run(['python', 'MathFeature/methods/Mappings-Protein.py',
	# 				'-n', str(len(fasta_list)), '-o',
	# 				dataset, '-r', '7'], text=True, input=text_input,
	# 				stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

	# with open(dataset, 'r') as temp_f:
	# 	col_count = [len(l.split(",")) for l in temp_f.readlines()]

	# colnames = ['EIIP_' + str(i) for i in range(0, max(col_count))]

	# df = pd.read_csv(dataset, names=colnames, header=None)
	# df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
	# df.to_csv(dataset, index=False)
	# datasets.append(dataset)

	# dataset = path + '/AAAF.csv'

	# subprocess.run(['python', 'MathFeature/methods/Mappings-Protein.py',
	# 				'-n', str(len(fasta_list)), '-o',
	# 				dataset, '-r', '1'], text=True, input=text_input,
	# 				stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

	# with open(dataset, 'r') as temp_f:
	# 	col_count = [len(l.split(",")) for l in temp_f.readlines()]

	# colnames = ['AccumulatedFrequency_' + str(i) for i in range(0, max(col_count))]

	# df = pd.read_csv(dataset, names=colnames, header=None)
	# df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
	# df.to_csv(dataset, index=False)
	# datasets.append(dataset)

	"""Concatenating all the extracted features"""
	
	if datasets:
		dfs_list = [
			pl.read_csv(f, ignore_errors=True)
			.select(pl.all().exclude("nameseq"), pl.col("nameseq"))
			.unique(subset=["nameseq"], keep='first')
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
	X_train.select(pl.all().exclude(["category", "nameseq", "label"])).write_csv(ftrain)
	
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
		X_test.select(pl.all().exclude(["category", "nameseq", "label"])).write_csv(ftest)

	return fnameseqtest, ftrain, flabeltrain, ftest, flabeltest

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
##                         Empowering Researchers with Machine Learning                           ##
##                                                                                                ##
##                                      Protein module                                            ##
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
	parser.add_argument('-algorithm', '--algorithm', default=0, help='0 - Bayesian Optimization ---- 1 - Genetic Algorithm')
	parser.add_argument('-imbalance', '--imbalance', default=0, help='Imbalanced data methods - 0: False, 1: True - Default: False')
	parser.add_argument('-fselection', '--fselection', default=0, help='Feature selection - 0: False, 1: True - Default: False')
	parser.add_argument('-estimations', '--estimations', default=50, help='number of estimations - BioAutoML - default = 50')
	parser.add_argument('-n_cpu', '--n_cpu', default=-1, help='number of cpus - default = all')
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')

	args = parser.parse_args()
	fasta_train = args.fasta_train
	fasta_label_train = args.fasta_label_train
	fasta_test = args.fasta_test
	fasta_label_test = args.fasta_label_test
	algo = int(args.algorithm)
	estimations = int(args.estimations)
	imbalance_data = args.imbalance
	fs = args.fselection
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

	# features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

	# process = multiprocessing.Process(target=feature_extraction, args=(fasta_train, 
    #                                                                   fasta_label_train,
    #                                                                   fasta_test, 
    #                                                                   fasta_label_test, 
    #                                                                   features, foutput))
	# print(process)
	# process.start()
	# process.join()

	fnameseqtest, ftrain, ftrain_labels, \
		ftest, ftest_labels = feature_extraction(fasta_train, fasta_label_train,
												 fasta_test, fasta_label_test, foutput)

	if algo == 0:
		classifier, path_train, path_test, train_best, test_best = \
          feature_engineering_optuna(estimations, ftrain, ftrain_labels, ftest, foutput)
	else:
		classifier, path_train, path_test, train_best, test_best = \
          feature_engineering_pygad(estimations, ftrain, ftrain_labels, ftest, foutput)
    
	# classifier, path_train, path_test, train_best, test_best = \
    #  	feature_engineering_ga_sklearn(ftrain, ftrain_labels, ftest, foutput)
       
	cost = (time.time() - start_time) / 60
	print('Computation time - Pipeline - Automated Feature Engineering: %s minutes' % cost)

	if len(fasta_label_train) > 2:
		subprocess.run(['python', 'BioAutoML-multiclass.py', '-train', path_train,
						 '-train_label', ftrain_labels, '-test', path_test,
						 '-test_label', ftest_labels, '-test_nameseq',
						 fnameseqtest, '-nf', 'True', '-fselection', fs,  
       					 '-imbalance', imbalance_data, '-n_cpu', str(n_cpu), 
       					 '-classifier', str(classifier), '-output', foutput])
	else:
		subprocess.run(['python', 'BioAutoML-binary.py', '-train', path_train,
						 '-train_label', ftrain_labels, '-test', path_test, '-test_label',
						 ftest_labels, '-test_nameseq', fnameseqtest,
						 '-nf', 'True', '-fselection', fs,  
       					 '-imbalance', imbalance_data, '-classifier', str(classifier),
						  '-n_cpu', str(n_cpu), '-output', foutput])

##########################################################################
##########################################################################
