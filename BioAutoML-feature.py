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
from genetic_selection import GeneticSelectionCV
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
from sklearn_genetic import GAFeatureSelectionCV
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, SparkTrials, early_stop
from subprocess import Popen
from multiprocessing import Manager


def objective_rf(space):

	"""Automated Feature Engineering - Objective Function - Bayesian Optimization"""

	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464)), 'repDNA': list(range(464, len(df_x.columns)))}

	for descriptor, ind in descriptors.items():
		if int(space[descriptor]) == 1:
			index = index + ind

	x = df_x[:, index]

	print(space)

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
	else:
		sys.exit('This classifier option does not exist - Try again')


	if len(fasta_label_train) > 2:
		score = make_scorer(f1_score, average='weighted')
	else:
		score = make_scorer(balanced_accuracy_score)

	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	le = LabelEncoder()
	metric = cross_val_score(model,
							 x,
        					 le.fit_transform(labels_y),
							 cv=kfold,
							 scoring=score,
							 n_jobs=n_cpu).mean()

	return {'loss': -metric, 'status': STATUS_OK}


def feature_engineering(estimations, train, train_labels, test, foutput):

	"""Automated Feature Engineering - Bayesian Optimization"""

	global df_x, labels_y

	print('Automated Feature Engineering - Bayesian Optimization')

	df_x = pl.read_csv(train)
	labels_y = pl.read_csv(train_labels)
 	#.values.ravel()

	if test != '':
		df_test = pl.read_csv(test)

	path_bio = foutput + '/best_descriptors'
	if not os.path.exists(path_bio):
		os.mkdir(path_bio)

	param = {'NAC': [0, 1], 'DNC': [0, 1],
			 'TNC': [0, 1], 'kGap_di': [0, 1], 'kGap_tri': [0, 1],
			 'ORF': [0, 1], 'Fickett': [0, 1],
			 'Shannon': [0, 1], 'FourierBinary': [0, 1],
			 'FourierComplex': [0, 1], 'Tsallis': [0, 1],
			 'repDNA': [0, 1],
			 'Classifier': [1, 2, 3]}

	space = {'NAC': hp.choice('NAC', [0, 1]),
			 'DNC': hp.choice('DNC', [0, 1]),
			 'TNC': hp.choice('TNC', [0, 1]),
			 'kGap_di': hp.choice('kGap_di', [0, 1]),
			 'kGap_tri': hp.choice('kGap_tri', [0, 1]),
			 'ORF': hp.choice('ORF', [0, 1]),
			 'Fickett': hp.choice('Fickett', [0, 1]),
			 'Shannon': hp.choice('Shannon', [0, 1]),
			 'FourierBinary': hp.choice('FourierBinary', [0, 1]),
			 'FourierComplex': hp.choice('FourierComplex', [0, 1]),
			 'Tsallis': hp.choice('Tsallis', [0, 1]),
   	 		 'repDNA': hp.choice('repDNA', [0, 1]),
			 'Classifier': hp.choice('Classifier', [1, 2, 3])}

	# spark_trials = SparkTrials(parallelism=n_cpu, timeout=7200)
	trials = Trials()
	best_tuning = fmin(fn=objective_rf,
				space=space,
				algo=tpe.suggest,
				max_evals=estimations,
				early_stop_fn=early_stop,
				trials=trials)

	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464)), 'repDNA': list(range(464, len(df_x.columns)))}

	for descriptor, ind in descriptors.items():
		result = param[descriptor][best_tuning[descriptor]]
		if result == 1:
			index = index + ind

	classifier = param['Classifier'][best_tuning['Classifier']]

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
 
 
def feature_engineering_ga(train, train_labels, test, foutput):
    
    """Automated Feature Engineering - Genetic Algorithm"""
    
    df_x = pl.read_csv(train)
    labels_y = pl.read_csv(train_labels)
    # .values.ravel()
    le = LabelEncoder()
    
    if test != '':
        df_test = pl.read_csv(test)
    
    path_bio = foutput + '/best_descriptors'
    if not os.path.exists(path_bio):
        os.mkdir(path_bio)
        
    # print("Selecting features with genetic algorithm...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    score = make_scorer(balanced_accuracy_score)
    classifier = best_algorithms(df_x, le.fit_transform(labels_y))
    # print(classifier)
    if classifier == 0:
        model = CatBoostClassifier(thread_count=n_cpu, nan_mode='Max',
                                   logging_level='Silent', random_state=63)
    elif classifier == 1:
        model = RandomForestClassifier(n_jobs=n_cpu, random_state=63)
        
    elif classifier == 2:
        model = lgb.LGBMClassifier(n_jobs=n_cpu, random_state=63)
    
    else:
        model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=n_cpu, random_state=63)
            
    selection = GeneticSelectionCV(estimator=model,
                                   cv=cv,
                                   scoring=score,
                                   n_population=10,
                                   verbose=1, 
                                   n_jobs=n_cpu,
                                   crossover_proba=0.5,
                                   mutation_proba=0.2,
                                   n_generations=50,
                                   crossover_independent_proba=0.5,
                                   mutation_independent_proba=0.05,
                                   tournament_size=3,
                                   n_gen_no_change=5)
    
    selection.fit(df_x, le.fit_transform(labels_y))
    
    features = selection.support_
    index = []
    # print(len(features))
    
    for i in range(0, len(features)):
        print(features[i])
        if str(features[i]) == 'True':
            index.append(i)
            
    # print(index)
    
    # btrain = selection.transform(df_x)
    btrain = df_x[:, index]
    path_btrain = path_bio + '/best_train.csv'
    btrain.write_csv(path_btrain, index=False, header=True)
    
    if test != '':
        # btest = selection.transform(df_test)
        btest = df_test[:, index]
        path_btest = path_bio + '/best_test.csv'
        btest.write_csv(path_btest, index=False, header=True)
    else:
        btest, path_btest = '', ''

	
    return classifier, path_btrain, path_btest, btrain, btest


def feature_engineering_ga_sklearn(train, train_labels, test, foutput):
    
    """Automated Feature Engineering - Genetic Algorithm"""
    
    df_x = pl.read_csv(train)
    labels_y = pl.read_csv(train_labels)
    # .values.ravel()
    le = LabelEncoder()
    
    if test != '':
        df_test = pl.read_csv(test)
    
    path_bio = foutput + '/best_descriptors'
    if not os.path.exists(path_bio):
        os.mkdir(path_bio)
        
    print("Selecting features with genetic algorithm...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    score = make_scorer(balanced_accuracy_score)
    classifier = best_algorithms(df_x, le.fit_transform(labels_y))
    # print(classifier)
    if classifier == 0:
        model = CatBoostClassifier(thread_count=n_cpu, nan_mode='Max',
                                   logging_level='Silent', random_state=63)
    elif classifier == 1:
        model = RandomForestClassifier(n_jobs=n_cpu, random_state=63)
        
    elif classifier == 2:
        model = lgb.LGBMClassifier(n_jobs=n_cpu, random_state=63)
    
    else:
        model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=n_cpu, random_state=63)
        
    selection = GAFeatureSelectionCV(estimator=model,
                                   cv=cv,
                                   scoring=score,
                                   population_size=20,
                                   generations=50, 
                                   n_jobs=n_cpu,
                                   verbose=True,
                                   keep_top_k=4,
                                   elitism=True)
    
    selection.fit(df_x, le.fit_transform(labels_y))
    
    features = selection.support_
    index = []
    # print(len(features))
    
    for i in range(0, len(features)):
        print(features[i])
        if str(features[i]) == 'True':
            index.append(i)
            
    # print(index)
    
    # btrain = selection.transform(df_x)
    btrain = df_x[:, index]
    path_btrain = path_bio + '/best_train.csv'
    btrain.write_csv(path_btrain, index=False, header=True)
    
    if test != '':
        # btest = selection.transform(df_test)
        btest = df_test[:, index]
        path_btest = path_bio + '/best_test.csv'
        btest.write_csv(path_btest, index=False, header=True)
    else:
        btest, path_btest = '', ''

	
    return classifier, path_btrain, path_btest, btrain, btest


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

	space = {'NAC': trial.suggest_categorical('NAC', [0, 1]),
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
   	 		 'repDNA': trial.suggest_categorical('repDNA', [0, 1]),
			 'Classifier': trial.suggest_categorical('Classifier', [1, 2, 3])}
	
	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464)), 'repDNA': list(range(464, 734))}
 
 
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


	param = {'NAC': [0, 1], 'DNC': [0, 1],
			 'TNC': [0, 1], 'kGap_di': [0, 1], 'kGap_tri': [0, 1],
			 'ORF': [0, 1], 'Fickett': [0, 1],
			 'Shannon': [0, 1], 'FourierBinary': [0, 1],
			 'FourierComplex': [0, 1], 'Tsallis': [0, 1],
			 'repDNA': [0, 1],
			 'Classifier': [1, 2, 3]}

	func = lambda trial: objective(trial, ns.df, train_labels)
	
	results = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
	results.optimize(func, n_trials=estimations, timeout=7200, n_jobs=n_cpu, show_progress_bar=True)
 
	best_tuning = results.best_params
 
	print(best_tuning)
	
	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464)), 'repDNA': list(range(464, 734))}

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
			datasets.append(path + '/NAC.csv')
			datasets.append(path + '/DNC.csv')
			datasets.append(path + '/TNC.csv')
			datasets.append(path + '/kGap_di.csv')
			datasets.append(path + '/kGap_tri.csv')
			datasets.append(path + '/ORF.csv')
			datasets.append(path + '/Fickett.csv')
			datasets.append(path + '/Shannon.csv')
			datasets.append(path + '/FourierBinary.csv')
			datasets.append(path + '/FourierComplex.csv')
			datasets.append(path + '/Tsallis.csv')
			datasets.append(path + '/repDNA.csv')
   
			commands = [['python', 'MathFeature/methods/ExtractionTechniques.py',
								'-i', preprocessed_fasta, '-o', path + '/NAC.csv', '-l', labels[i][j],
								'-t', 'NAC', '-seq', '1'],

						['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
								preprocessed_fasta, '-o', path + '/DNC.csv', '-l', labels[i][j],
								'-t', 'DNC', '-seq', '1'],
      
						['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
								preprocessed_fasta, '-o', path + '/TNC.csv', '-l', labels[i][j],
								'-t', 'TNC', '-seq', '1'],
						['python', 'MathFeature/methods/Kgap.py', '-i',
								preprocessed_fasta, '-o', path + '/kGap_di.csv', '-l',
								labels[i][j], '-k', '1', '-bef', '1',
								'-aft', '2', '-seq', '1'],
						['python', 'MathFeature/methods/Kgap.py', '-i',
								preprocessed_fasta, '-o', path + '/kGap_tri.csv', '-l',
								labels[i][j], '-k', '1', '-bef', '1',
								'-aft', '3', '-seq', '1'],
						['python', 'MathFeature/methods/CodingClass.py', '-i',
								preprocessed_fasta, '-o', path + '/ORF.csv', '-l', labels[i][j]],
						['python', 'MathFeature/methods/FickettScore.py', '-i',
								preprocessed_fasta, '-o', path + '/Fickett.csv', '-l', labels[i][j],
								'-seq', '1'],
						['python', 'MathFeature/methods/EntropyClass.py', '-i',
								preprocessed_fasta, '-o', path + '/Shannon.csv', '-l', labels[i][j],
								'-k', '5', '-e', 'Shannon'],
						['python', 'MathFeature/methods/FourierClass.py', '-i',
								preprocessed_fasta, '-o', path + '/FourierBinary.csv', '-l', labels[i][j],
								'-r', '1'],
						['python', 'other-methods/FourierClass.py', '-i',
								preprocessed_fasta, '-o', path + '/FourierComplex.csv', '-l', labels[i][j],
								'-r', '6'],
						['python', 'other-methods/TsallisEntropy.py', '-i',
								preprocessed_fasta, '-o', path + '/Tsallis.csv', '-l', labels[i][j],
								'-k', '5', '-q', '2.3'],
						['python', 'other-methods/repDNA/repDNA-feat.py', '--file',
								preprocessed_fasta, '--output', path + '/repDNA.csv', '--label', labels[i][j]]
			]

			processes = [Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) for cmd in commands]
			for p in processes: p.wait()

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
