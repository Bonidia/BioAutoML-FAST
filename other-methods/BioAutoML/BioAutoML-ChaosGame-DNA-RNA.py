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
import pygad
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from subprocess import Popen
from multiprocessing import Manager

 
def check_techniques(model, train, train_labels):
    """Testing algorithms"""

    kfold = StratifiedKFold(n_splits=2, shuffle=True)
    acc = cross_val_score(model,
                          train,
                          train_labels,
                          cv=kfold,
                          scoring=make_scorer(balanced_accuracy_score),
                          n_jobs=n_cpu).mean()
    return acc


def best_algorithms(train, train_labels):

    print('Checking the best algorithms...')
    performance = []
    # cata = CatBoostClassifier(thread_count=n_cpu, nan_mode='Max',
    #                         logging_level='Silent', random_state=63)
    rfa = RandomForestClassifier(n_jobs=n_cpu, random_state=63)
    lgba = lgb.LGBMClassifier(n_jobs=n_cpu, random_state=63)
    xgba = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=n_cpu, random_state=63)
    # one = check_techniques(cata, train, train_labels)
    two = check_techniques(rfa, train, train_labels)
    three = check_techniques(lgba, train, train_labels)
    four = check_techniques(xgba, train, train_labels)
    # performance.append(one)
    performance.append(two)
    performance.append(three)
    performance.append(four)
    max_pos = performance.index(max(performance))
    return max_pos + 1


def objective_ga_pygad(ga_instance, solution, solution_idx):

	"""Automated Feature Engineering - Objective Function - Genetic Algorithm"""
	
	index = list()
	for gene in range(0, len(solution)):
		if int(solution[gene]) == 1:
			index.append(int(gene))
	# print(index)
	
 
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

	print('Checking the best features...')
	ga_instance = pygad.GA(num_generations=estimations,
                       num_parents_mating=4,
                       fitness_func=objective_ga_pygad,
                       sol_per_pop=20,
                       num_genes=len(df_x.columns),
                       gene_type=int,
                       init_range_low=0,
                       init_range_high=2,
                       parent_selection_type="tournament",
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
	for gene in range(0, len(best)):
		if int(best[gene]) == 1:
			index.append(int(gene))
   
	# print(index)

	if test != '':
		df_test = pl.read_csv(test)

	# print(index)
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
				subprocess.run(['python', 'MathFeature/preprocessing/preprocessing.py',
								'-i', fasta[i][j], '-o', preprocessed_fasta],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				train_size += len([1 for line in open(preprocessed_fasta) if line.startswith(">")])
			else:  # Test
				preprocessed_fasta = path + '/test/pre_' + file
				subprocess.run(['python', 'MathFeature/preprocessing/preprocessing.py',
								'-i', fasta[i][j], '-o', preprocessed_fasta],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

			fasta_list.append(preprocessed_fasta)
			
	dataset = path + '/Chaos.csv'
	labels_list = ftrain_labels
	if fasta_test:
		labels_list = ftrain_labels + ftest_labels
	text_input = ''
	for i in range(len(fasta_list)):
		text_input += fasta_list[i] + '\n' + labels_list[i] + '\n'

	subprocess.run(['python', 'MathFeature/methods/ChaosGameTheory.py',
                   	'-n', str(len(fasta_list)), '-o',
					dataset, '-r', '1'], text=True, input=text_input,
				   	stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

	with open(dataset, 'r') as temp_f:
			col_count = [len(l.split(",")) for l in temp_f.readlines()]

	colnames = ['ChaosMapping_' + str(i) for i in range(0, max(col_count))]
	df = pd.read_csv(dataset, names=colnames, header=None)
	df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
	df.to_csv(dataset, index=False)
	datasets.append(dataset)

	"""Concatenating all the extracted features"""

	if datasets:
		datasets = list(dict.fromkeys(datasets))
		dataframes = pd.concat([pd.read_csv(f) for f in datasets], axis=1)
		dataframes = dataframes.loc[:, ~dataframes.columns.duplicated()]
		dataframes = dataframes[~dataframes.nameseq.str.contains("nameseq")]

	X_train = dataframes.iloc[:train_size, :]
	X_train.pop('nameseq')
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
##                                      ChaosGame module                                          ##
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
	parser.add_argument('-estimations', '--estimations', default=100, help='number of estimations - BioAutoML - default = 50')
	parser.add_argument('-n_cpu', '--n_cpu', default=-1, help='number of cpus - default = all')
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')

	args = parser.parse_args()
	fasta_train = args.fasta_train
	fasta_label_train = args.fasta_label_train
	fasta_test = args.fasta_test
	fasta_label_test = args.fasta_label_test
	algo = int(args.algorithm)
	estimations = int(args.estimations)
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


	fnameseqtest, ftrain, ftrain_labels, \
		ftest, ftest_labels = feature_extraction(fasta_train, fasta_label_train,
												 fasta_test, fasta_label_test, foutput)

	classifier, path_train, path_test, train_best, test_best = \
          feature_engineering_pygad(estimations, ftrain, ftrain_labels, ftest, foutput)
       
	cost = (time.time() - start_time) / 60
	print('Computation time - Pipeline - Automated Feature Engineering: %s minutes' % cost)

	if len(fasta_label_train) > 2:
		subprocess.run(['python', 'BioAutoML-multiclass.py', '-train', path_train,
						 '-train_label', ftrain_labels, '-test', path_test,
						 '-test_label', ftest_labels, '-test_nameseq',
						 fnameseqtest, '-nf', 'True', '-n_cpu', str(n_cpu), 
       					 '-classifier', str(classifier), '-output', foutput])
	else:
		subprocess.run(['python', 'BioAutoML-binary.py', '-train', path_train,
						 '-train_label', ftrain_labels, '-test', path_test, '-test_label',
						 ftest_labels, '-test_nameseq', fnameseqtest,
						 '-nf', 'True', '-fs', str(0), '-classifier', str(classifier), 
       					 '-imbalance', 'True', '-n_cpu', str(n_cpu), '-output', foutput])

##########################################################################
##########################################################################
