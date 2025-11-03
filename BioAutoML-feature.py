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
import numpy as np
from genetic_selection import GeneticSelectionCV
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn_genetic import GAFeatureSelectionCV
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, SparkTrials, early_stop
from subprocess import Popen
from multiprocessing import Manager
from Bio import SeqIO

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
		model = lgb.LGBMClassifier(n_jobs=1, random_state=63, verbosity=-1)
	elif int(space['Classifier']) == 3:
		model = xgb.XGBClassifier(eval_metric='mlogloss', n_jobs=1, random_state=63)
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
    lgba = lgb.LGBMClassifier(n_jobs=n_cpu, random_state=63, verbosity=-1)
    xgba = xgb.XGBClassifier(eval_metric='mlogloss', n_jobs=n_cpu, random_state=63)
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
        model = lgb.LGBMClassifier(n_jobs=n_cpu, random_state=63, verbosity=-1)
    
    else:
        model = xgb.XGBClassifier(eval_metric='mlogloss', n_jobs=n_cpu, random_state=63)
            
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
    btrain.write_csv(path_btrain, index=False)
    
    if test != '':
        # btest = selection.transform(df_test)
        btest = df_test[:, index]
        path_btest = path_bio + '/best_test.csv'
        btest.write_csv(path_btest, index=False)
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
        model = lgb.LGBMClassifier(n_jobs=n_cpu, random_state=63, verbosity=-1)
    
    else:
        model = xgb.XGBClassifier(eval_metric='mlogloss', n_jobs=n_cpu, random_state=63)
        
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

def feature_engineering_pygad(task, estimations, fnameseqtrain, train, train_labels, test, foutput):
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
        model = lgb.LGBMClassifier(n_jobs=1, random_state=63, verbosity=-1)
    else:
        model = xgb.XGBClassifier(eval_metric='mlogloss', n_jobs=1, random_state=63)

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
    print("Best fitness value reached after {best_solution_generation} generations.".format(
        best_solution_generation=ga_instance.best_solution_generation))
 
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

def objective(trial, train, task, y):
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
        'repDNA': trial.suggest_categorical('repDNA', [0, 1]),
        'Classifier': trial.suggest_categorical('Classifier', [1, 2, 3])
    }

    # Descriptor indices
    descriptors = {
        'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
        'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
        'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
        'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
        'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
        'Tsallis': list(range(459, 464)), 'repDNA': list(range(464, 734))
    }

    index = []
    for descriptor, ind in descriptors.items():
        if int(space[descriptor]) == 1:
            index.extend(ind)

    # === Classification Task ===
    if task == 0:
        if space['Classifier'] == 0:
            model = CatBoostClassifier(nan_mode='Max', logging_level='Silent', random_state=63)
        elif space['Classifier'] == 1:
            model = RandomForestClassifier(random_state=63)
        elif space['Classifier'] == 2:
            model = lgb.LGBMClassifier(random_state=63, verbosity=-1)
        elif space['Classifier'] == 3:
            model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=63)

        # Use weighted F1 for multiclass, balanced accuracy for binary
        if len(np.unique(y)) > 2:
            score = make_scorer(f1_score, average='weighted')
        else:
            score = make_scorer(balanced_accuracy_score)

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)

    # === Regression Task ===
    elif task == 1:
        if space['Classifier'] == 0:
            model = CatBoostRegressor(nan_mode='Max', logging_level='Silent', random_state=63)
        elif space['Classifier'] == 1:
            model = RandomForestRegressor(random_state=63)
        elif space['Classifier'] == 2:
            model = lgb.LGBMRegressor(random_state=63, verbosity=-1)
        elif space['Classifier'] == 3:
            model = xgb.XGBRegressor(random_state=63)

        score = make_scorer(r2_score)
        kfold = KFold(n_splits=2, shuffle=True, random_state=63)
    else:
        raise ValueError("Invalid task type. Use 0 for classification or 1 for regression.")

    # === Cross-validation ===
    try:
        metric = cross_val_score(
            model,
            train.iloc[:, index],
            y,
            cv=kfold,
            scoring=score
        ).mean()
    except Exception:
        raise optuna.TrialPruned()

    return metric

def feature_engineering_optuna(task, estimations, fnameseqtrain, train, train_labels, test, foutput):
    """Automated Feature Engineering - Bayesian Optimization"""
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
             'Shannon': [0, 1], 'FourierBinary': [0, 1],
             'FourierComplex': [0, 1], 'Tsallis': [0, 1],
             'repDNA': [0, 1],
             'Classifier': [1, 2, 3]}
    
    if task == 0:
        labels = pd.read_csv(train_labels)
        le = LabelEncoder()
        y = le.fit_transform(labels)
    elif task == 1:
        y = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(fnameseqtrain)["nameseq"].to_list()]

    func = lambda trial: objective(trial, ns.df, task, y)
    
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
##                         Empowering Researchers with Machine Learning                           ##
##                                                                                                ##
##                                      DNA/RNA module                                            ##
##                                                                                                ##
####################################################################################################
####################################################################################################
    ''')
    parser = argparse.ArgumentParser()
    parser.add_argument('-fasta_train', '--fasta_train', nargs='+',
                        help='fasta format file, e.g., training/ncRNA.fasta'
                                'training/lncRNA.fasta training/circRNA.fasta')
    parser.add_argument('-fasta_label_train', '--fasta_label_train', nargs='+',
                        help='labels for fasta files, e.g., ncRNA lncRNA circRNA')
    parser.add_argument('-fasta_test', '--fasta_test', nargs='+',
                        help='fasta format file, e.g., testing/ncRNA.fasta testing/lncRNA.fasta testing/circRNA.fasta')
    parser.add_argument('-fasta_label_test', '--fasta_label_test', nargs='+',
                        help='labels for fasta files, e.g., ncRNA lncRNA circRNA')
    parser.add_argument('-algorithm', '--algorithm', default=0, help='Optimization algorithm - 0: Bayesian Optimization, 1: Genetic Algorithm - Default: 0')
    parser.add_argument('-task', '--task', default=0, help='Machine learning task - 0: Classification, 1: Regression - Default: Classification')
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
    task = int(args.task)
    estimations = int(args.estimations)
    imbalance_data = str(args.imbalance)
    fs = str(args.fselection)
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
        fnameseqtrain, fnameseqtest, ftrain, ftrain_labels, \
            ftest, ftest_labels = feature_extraction(fasta_train, fasta_label_train,
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
         
    if algo == 0:
        classifier, path_train, path_test, train_best, test_best = \
            feature_engineering_optuna(task, estimations, fnameseqtrain, ftrain, ftrain_labels, ftest, foutput)
    else:
        classifier, path_train, path_test, train_best, test_best = \
            feature_engineering_pygad(task, estimations, fnameseqtrain, ftrain, ftrain_labels, ftest, foutput)

    # classifier, path_train, path_test, train_best, test_best = \
    #  	feature_engineering_ga_sklearn(ftrain, ftrain_labels, ftest, foutput)
        
    cost = (time.time() - start_time) / 60
    print('Computation time - Pipeline - Automated Feature Engineering: %s minutes' % cost)

    # if len(fasta_label_train) > 2:
    #     subprocess.run(['python', 'BioAutoML-multiclass.py', '-train', path_train,
    #                         '-train_label', ftrain_labels, '-test', path_test,
    #                         '-test_label', ftest_labels, '-train_nameseq', fnameseqtrain,
    #                         '-test_nameseq', fnameseqtest, '-nf', 'True', '-fselection', fs,  
    #                         '-imbalance', imbalance_data, '-n_cpu', str(n_cpu), 
    #                         '-classifier', str(classifier), '-output', foutput])
    # else:
    #     subprocess.run(['python', 'BioAutoML-binary.py', '-train', path_train,
    #                         '-train_label', ftrain_labels, '-test', path_test, 
    #                         '-test_label', ftest_labels, '-train_nameseq', fnameseqtrain,
    #                         '-test_nameseq', fnameseqtest, '-nf', 'True', '-fselection', fs,  
    #                         '-imbalance', imbalance_data, '-classifier', str(classifier), 
    #                         '-n_cpu', str(n_cpu), '-output', foutput])

    ##########################################################################
    ##########################################################################
