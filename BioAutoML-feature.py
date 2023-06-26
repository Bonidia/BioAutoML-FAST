import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
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

	x = df_x.iloc[:, index]

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

	df_x = pd.read_csv(train)
	labels_y = pd.read_csv(train_labels).values.ravel()

	if test != '':
		df_test = pd.read_csv(test)

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

	btrain = df_x.iloc[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.to_csv(path_btrain, index=False, header=True)

	if test != '':
		btest = df_test.iloc[:, index]
		path_btest = path_bio + '/best_test.csv'
		btest.to_csv(path_btest, index=False, header=True)
	else:
		btest, path_btest = '', ''

	return classifier, path_btrain, path_btest, btrain, btest
 
 
def check_techniques(model, train, train_labels):
    """Testing algorithms"""

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
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
    cata = CatBoostClassifier(thread_count=n_cpu, nan_mode='Max',
                             logging_level='Silent', random_state=63)
    rfa = RandomForestClassifier(n_jobs=n_cpu, random_state=63)
    lgba = lgb.LGBMClassifier(n_jobs=n_cpu, random_state=63)
    xgba = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=n_cpu, random_state=63)
    one = check_techniques(cata, train, train_labels)
    two = check_techniques(rfa, train, train_labels)
    three = check_techniques(lgba, train, train_labels)
    four = check_techniques(xgba, train, train_labels)
    performance.append(one)
    performance.append(two)
    performance.append(three)
    performance.append(four)
    max_pos = performance.index(max(performance))
    return max_pos
 
 
def feature_engineering_ga(train, train_labels, test, foutput):
    
    """Automated Feature Engineering - Genetic Algorithm"""
    
    df_x = pd.read_csv(train)
    labels_y = pd.read_csv(train_labels).values.ravel()
    le = LabelEncoder()
    
    if test != '':
        df_test = pd.read_csv(test)
    
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
    btrain = df_x.iloc[:, index]
    path_btrain = path_bio + '/best_train.csv'
    btrain.to_csv(path_btrain, index=False, header=True)
    
    if test != '':
        # btest = selection.transform(df_test)
        btest = df_test.iloc[:, index]
        path_btest = path_bio + '/best_test.csv'
        btest.to_csv(path_btest, index=False, header=True)
    else:
        btest, path_btest = '', ''

	
    return classifier, path_btrain, path_btest, btrain, btest


def feature_engineering_ga_sklearn(train, train_labels, test, foutput):
    
    """Automated Feature Engineering - Genetic Algorithm"""
    
    df_x = pd.read_csv(train)
    labels_y = pd.read_csv(train_labels).values.ravel()
    le = LabelEncoder()
    
    if test != '':
        df_test = pd.read_csv(test)
    
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
    btrain = df_x.iloc[:, index]
    path_btrain = path_bio + '/best_train.csv'
    btrain.to_csv(path_btrain, index=False, header=True)
    
    if test != '':
        # btest = selection.transform(df_test)
        btest = df_test.iloc[:, index]
        path_btest = path_bio + '/best_test.csv'
        btest.to_csv(path_btest, index=False, header=True)
    else:
        btest, path_btest = '', ''

	
    return classifier, path_btrain, path_btest, btrain, btest


def objective(trial):

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
				   'Tsallis': list(range(459, 464)), 'repDNA': list(range(464, len(df_x.columns)))}

	for descriptor, ind in descriptors.items():
		if int(space[descriptor]) == 1:
			index = index + ind

	x = df_x.iloc[:, index]

	# print(space)

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
	metric = cross_val_score(model,
							 x,
        					 le.fit_transform(labels_y),
							 cv=kfold,
							 scoring=score,
							 n_jobs=n_cpu).mean()

	return metric


def feature_engineering_optuna(estimations, train, train_labels, test, foutput):

	"""Automated Feature Engineering - Bayesian Optimization"""

	global df_x, labels_y

	print('Automated Feature Engineering - Bayesian Optimization')

	df_x = pd.read_csv(train)
	labels_y = pd.read_csv(train_labels).values.ravel()

	if test != '':
		df_test = pd.read_csv(test)

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

	results = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
	results.optimize(objective, n_trials=estimations, timeout=7200, n_jobs=n_cpu, show_progress_bar=True)
 
	best_tuning = results.best_params
 
	print(best_tuning)
 
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

	classifier = best_tuning['Classifier']
	# print(classifier)

	btrain = df_x.iloc[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.to_csv(path_btrain, index=False, header=True)

	if test != '':
		btest = df_test.iloc[:, index]
		path_btest = path_bio + '/best_test.csv'
		btest.to_csv(path_btest, index=False, header=True)
	else:
		btest, path_btest = '', ''

	return classifier, path_btrain, path_btest, btrain, btest


def feature_extraction(ftrain, ftrain_labels, ftest, ftest_labels, features, foutput):

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
 
	# commands = [
	# 	['python', 'MathFeature/preprocessing/preprocessing.py',
   	# 	 '-i', fasta[i][j], '-o', preprocessed_fasta]
	# ]

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

			if 1 in features:
				dataset = path + '/NAC.csv'
				subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py',
								'-i', preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-t', 'NAC', '-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 2 in features:
				dataset = path + '/DNC.csv'
				subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-t', 'DNC', '-seq', '1'], stdout=subprocess.DEVNULL,
								stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 3 in features:
				dataset = path + '/TNC.csv'
				subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-t', 'TNC', '-seq', '1'], stdout=subprocess.DEVNULL,
								stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 4 in features:
				dataset_di = path + '/kGap_di.csv'
				dataset_tri = path + '/kGap_tri.csv'

				subprocess.run(['python', 'MathFeature/methods/Kgap.py', '-i',
								preprocessed_fasta, '-o', dataset_di, '-l',
								labels[i][j], '-k', '1', '-bef', '1',
								'-aft', '2', '-seq', '1'],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

				subprocess.run(['python', 'MathFeature/methods/Kgap.py', '-i',
								preprocessed_fasta, '-o', dataset_tri, '-l',
								labels[i][j], '-k', '1', '-bef', '1',
								'-aft', '3', '-seq', '1'],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset_di)
				datasets.append(dataset_tri)

			if 5 in features:
				dataset = path + '/ORF.csv'
				subprocess.run(['python', 'MathFeature/methods/CodingClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j]],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 6 in features:
				dataset = path + '/Fickett.csv'
				subprocess.run(['python', 'MathFeature/methods/FickettScore.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 7 in features:
				dataset = path + '/Shannon.csv'
				subprocess.run(['python', 'MathFeature/methods/EntropyClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-k', '5', '-e', 'Shannon'],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 8 in features:
				dataset = path + '/FourierBinary.csv'
				subprocess.run(['python', 'MathFeature/methods/FourierClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-r', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 9 in features:
				dataset = path + '/FourierComplex.csv'
				subprocess.run(['python', 'other-methods/FourierClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-r', '6'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 10 in features:
				dataset = path + '/Tsallis.csv'
				subprocess.run(['python', 'other-methods/TsallisEntropy.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-k', '5', '-q', '2.3'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)
    
			if 11 in features:
				dataset = path + '/repDNA.csv'
				subprocess.run(['python', 'other-methods/repDNA/repDNA-feat.py', '--file',
								preprocessed_fasta, '--output', dataset, '--label', labels[i][j]],
                   				stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

	if 12 in features:
		dataset = path + '/Chaos.csv'
		# classifical_chaos(preprocessed_fasta, labels[i][j], 'Yes', dataset)
		datasets.append(dataset)

	if 13 in features:
		dataset = path + '/BinaryMapping.csv'

		labels_list = ftrain_labels + ftest_labels
		text_input = ''
		for i in range(len(fasta_list)):
			text_input += fasta_list[i] + '\n' + labels_list[i] + '\n'

		subprocess.run(['python', 'MathFeature/methods/MappingClass.py',
						'-n', str(len(fasta_list)), '-o',
						dataset, '-r', '1'], text=True, input=text_input,
					   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

		with open(dataset, 'r') as temp_f:
			col_count = [len(l.split(",")) for l in temp_f.readlines()]

		colnames = ['BinaryMapping_' + str(i) for i in range(0, max(col_count))]

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

	return fnameseqtest, ftrain, flabeltrain, ftest, flabeltest

##########################################################################
##########################################################################


if __name__ == '__main__':
	print('\n')
	print('###################################################################################')
	print('###################################################################################')
	print('##########         BioAutoML- Automated Feature Engineering             ###########')
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
	parser.add_argument('-estimations', '--estimations', default=100, help='number of estimations - BioAutoML - default = 50')
	parser.add_argument('-n_cpu', '--n_cpu', default=-1, help='number of cpus - default = all')
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')

	args = parser.parse_args()
	fasta_train = args.fasta_train
	fasta_label_train = args.fasta_label_train
	fasta_test = args.fasta_test
	fasta_label_test = args.fasta_label_test
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

	features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

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
												 fasta_test, fasta_label_test, features, foutput)

	classifier, path_train, path_test, train_best, test_best = \
		feature_engineering_optuna(estimations, ftrain, ftrain_labels, ftest, foutput)
  
	# classifier, path_train, path_test, train_best, test_best = \
    #  	feature_engineering_ga(ftrain, ftrain_labels, ftest, foutput)
    
	# classifier, path_train, path_test, train_best, test_best = \
    #  	feature_engineering_ga_sklearn(ftrain, ftrain_labels, ftest, foutput)
       
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
