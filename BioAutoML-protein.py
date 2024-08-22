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
	df_x = pl.read_csv(train)
	y = le.fit_transform(pl.read_csv(train_labels))
	
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
		df_test = pl.read_csv(test)

	btrain = df_x[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.write_csv(path_btrain)

	if test != '':
		btest = df_test[:, index]
		path_btest = path_bio + '/best_test.csv'
		btest.write_csv(path_btest)
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

	df_x = pl.read_csv(train)
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
	# print(classifier)
	
 	# mem = sys.getsizeof(df_x)
	# print(mem)
	# max = 1073741824
	# if mem > max:
	# 	df_x = pl.read_csv(train).sample(n=(int(df_x.shape[0]*0.70)), seed=42)
	# else:
	# 	pass

	if test != '':
		df_test = pl.read_csv(test)

	btrain = ns.df[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.write_csv(path_btrain)

	if test != '':
		btest = df_test[:, index]
		path_btest = path_bio + '/best_test.csv'
		btest.write_csv(path_btest)
	else:
		btest, path_btest = '', ''

	return classifier, path_btrain, path_btest, btrain, btest


def feature_extraction(ftrain, ftrain_labels, ftest, ftest_labels, foutput):

	"""Extracts the features from the sequences in the fasta files."""

	path = foutput + '/feat_extraction'
	path_results = foutput

	try:
		shutil.rmtree(path)
		shutil.rmtree(path_results)
	except OSError as e:
		print("Error: %s - %s." % (e.filename, e.strerror))
		print('Creating Directory...')

	if not os.path.exists(path_results):
		os.mkdir(path_results)

	if not os.path.exists(path):
		os.mkdir(path)
		os.mkdir(path + '/train')
		os.mkdir(path + '/test')

	labels = [ftrain_labels]
	fasta = [ftrain]
	train_size = 0

	if fasta_test:
		labels.append(ftest_labels)
		fasta.append(ftest)

	datasets = []
	fasta_list = []

	print('Extracting features with MathFeature...')
 
	for i in range(len(labels)):
		for j in range(len(labels[i])):
			file = fasta[i][j].split('/')[-1]
			if i == 0:  # Train
				preprocessed_fasta = path + '/train/pre_' + file
				subprocess.run(['python', 'other-methods/preprocessing.py',
								'-i', fasta[i][j], '-o', preprocessed_fasta],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				train_size += len([1 for line in open(preprocessed_fasta) if line.startswith(">")])
			else:  # Test
				preprocessed_fasta = path + '/test/pre_' + file
				subprocess.run(['python', 'other-methods/preprocessing.py',
								'-i', fasta[i][j], '-o', preprocessed_fasta],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

			fasta_list.append(preprocessed_fasta)
			datasets.append(path + '/Shannon.csv')
			datasets.append(path + '/Tsallis_23.csv')
			datasets.append(path + '/Tsallis_30.csv')
			datasets.append(path + '/Tsallis_40.csv')
			datasets.append(path + '/ComplexNetworks.csv')
			datasets.append(path + '/kGap_di.csv')
			datasets.append(path + '/AAC.csv')
			datasets.append(path + '/DPC.csv')
			datasets.append(path + '/iFeature-features.csv')
			datasets.append(path + '/Global.csv')
			datasets.append(path + '/Peptide.csv')
			
			commands = [['python', 'MathFeature/methods/EntropyClass.py',
								'-i', preprocessed_fasta, '-o', path + '/Shannon.csv', '-l', labels[i][j],
								'-k', '5', '-e', 'Shannon'],

						['python', 'other-methods/TsallisEntropy.py',
								'-i', preprocessed_fasta, '-o', path + '/Tsallis_23.csv', '-l', labels[i][j],
								'-k', '5', '-q', '2.3'],
      
						['python', 'other-methods/TsallisEntropy.py',
								'-i', preprocessed_fasta, '-o', path + '/Tsallis_30.csv', '-l', labels[i][j],
								'-k', '5', '-q', '3.0'],
      
						['python', 'other-methods/TsallisEntropy.py',
								'-i', preprocessed_fasta, '-o', path + '/Tsallis_40.csv', '-l', labels[i][j],
								'-k', '5', '-q', '4.0'],
      
						['python', 'MathFeature/methods/ComplexNetworksClass-v2.py', '-i',
								preprocessed_fasta, '-o', path + '/ComplexNetworks.csv', '-l', labels[i][j],
								'-k', '3'],
      
						['python', 'MathFeature/methods/Kgap.py', '-i',
								preprocessed_fasta, '-o', path + '/kGap_di.csv', '-l',
								labels[i][j], '-k', '1', '-bef', '1',
								'-aft', '1', '-seq', '3'],
      
						['python', 'other-methods/ExtractionTechniques-Protein.py', '-i',
								preprocessed_fasta, '-o', path + '/AAC.csv', '-l', labels[i][j],
								'-t', 'AAC'],
      
						['python', 'other-methods/ExtractionTechniques-Protein.py', '-i',
								preprocessed_fasta, '-o', path + '/DPC.csv', '-l', labels[i][j],
								'-t', 'DPC'],
      
						['python', 'other-methods/iFeature-modified/iFeature.py', '--file',
								preprocessed_fasta, '--type', 'All', '--label', labels[i][j], 
        						'--out', path + '/iFeature-features.csv'],
      
						['python', 'other-methods/modlAMP-modified/descriptors.py', '-option',
								'global', '-label', labels[i][j], '-input', preprocessed_fasta, 
        						'-output', path + '/Global.csv'],
      
						['python', 'other-methods/modlAMP-modified/descriptors.py', '-option',
								'peptide', '-label', labels[i][j], '-input', preprocessed_fasta, 
        						'-output', path + '/Peptide.csv'],
			]

			processes = [Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) for cmd in commands]
			for p in processes: p.wait()

	dataset = path + '/Fourier_Integer.csv'
	if fasta_test:
		labels_list = ftrain_labels + ftest_labels
	else:
		labels_list = ftrain_labels
	text_input = ''
	for i in range(len(fasta_list)):
		text_input += fasta_list[i] + '\n' + labels_list[i] + '\n'

	subprocess.run(['python', 'MathFeature/methods/Mappings-Protein.py',
					'-n', str(len(fasta_list)), '-o',
					dataset, '-r', '6'], text=True, input=text_input,
					stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

	with open(dataset, 'r') as temp_f:
		col_count = [len(l.split(",")) for l in temp_f.readlines()]

	colnames = ['Integer_Fourier_' + str(i) for i in range(0, max(col_count))]

	df = pd.read_csv(dataset, names=colnames, header=0)
	df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
	df.to_csv(dataset, index=False)
	datasets.append(dataset)

	dataset = path + '/Fourier_EIIP.csv'

	subprocess.run(['python', 'MathFeature/methods/Mappings-Protein.py',
					'-n', str(len(fasta_list)), '-o',
					dataset, '-r', '8'], text=True, input=text_input,
					stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

	with open(dataset, 'r') as temp_f:
		col_count = [len(l.split(",")) for l in temp_f.readlines()]

	colnames = ['EIIP_Fourier_' + str(i) for i in range(0, max(col_count))]

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
		datasets = list(dict.fromkeys(datasets))
		dataframes = pd.concat([pd.read_csv(f) for f in datasets], axis=1)
		dataframes = dataframes.loc[:, ~dataframes.columns.duplicated()]
		dataframes = dataframes[~dataframes.nameseq.str.contains("nameseq")]

	X_train = dataframes.iloc[:train_size, :]
	nameseq_train = X_train.pop('nameseq')
	fnameseqtrain = path + '/fnameseqtrain.csv'
	nameseq_train.to_csv(fnameseqtrain, index=False, header=True)
	y_train = X_train.pop('label')
	ftrain = path + '/ftrain.csv'
	X_train.to_csv(ftrain, index=False)
	flabeltrain = path + '/flabeltrain.csv'
	y_train.to_csv(flabeltrain, index=False, header=True)
	
	fnameseqtest, ftest, flabeltest = '', '', ''

	if fasta_test:
		X_test = dataframes.iloc[train_size:, :]
		y_test = X_test.pop('label')
		nameseq_test = X_test.pop('nameseq')
		fnameseqtest = path + '/fnameseqtest.csv'
		nameseq_test.to_csv(fnameseqtest, index=False, header=True)
		ftest = path + '/ftest.csv'
		X_test.to_csv(ftest, index=False)
		flabeltest = path + '/flabeltest.csv'
		y_test.to_csv(flabeltest, index=False, header=True)
		del X_test

	del X_train
	return fnameseqtest, ftrain, flabeltrain, ftest, flabeltest

##########################################################################
##########################################################################


if __name__ == '__main__':
	print('\n')
	print('###################################################################################')
	print('###################################################################################')
	print('##########         BioAutoML-Fast: Automated Feature Engineering        ###########')
	print('##########              Author: Robson Parmezan Bonidia                 ###########')
	print('##########         WebPage: https://bonidia.github.io/website/          ###########')
	print('###################################################################################')
	print('###################################################################################')
	print('\n')
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
	parser.add_argument('-estimations', '--estimations', default=10, help='number of estimations - BioAutoML - default = 100')
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
