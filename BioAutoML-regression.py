import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
import argparse
import sys
import os.path
import time
import lightgbm as lgb
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score

def header(output_header):

	"""Header Function: Header of the evaluate_model_cross Function"""

	file = open(output_header, 'a')
	file.write('mean_absolute_error,std_mean_absolute_error,mean_squared_error'
			   ',std_mean_squared_error,root_mean_squared_error,std_root_mean_squared_error,'
			   'median_absolute_error,std_median_absolute_error,r2,std_r2')
	file.write('\n')
	return


def save_measures(output_measures, scores):

	"""Save Measures Function: Output of the evaluate_model_cross Function"""

	header(output_measures)
	file = open(output_measures, 'a')
	file.write('%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f' % (abs(scores['test_MAE']).mean(),
				+ abs(scores['test_MAE']).std(), abs(scores['test_MSE']).mean(),
				+ abs(scores['test_MSE']).std(),
				+ abs(scores['test_RMSE']).mean(), abs(scores['test_RMSE']).std(),
				+ abs(scores['test_MAE_median']).mean(), abs(scores['test_MAE_median']).std(),
				+ scores['test_R2'].mean(), scores['test_R2'].std()))
	file.write('\n')
	return


def evaluate_model_cross(X, y, model, output_cross):

	"""Evaluation Function: Using Cross-Validation"""

	scoring = {'MAE': 'neg_mean_absolute_error',
			   'MSE': 'neg_mean_squared_error',
			   'RMSE': 'neg_root_mean_squared_error',
			   'MAE_median': 'neg_median_absolute_error',
			   'R2': 'r2'}
	kfold = KFold(n_splits=5, shuffle=True)
	scores = cross_validate(model, X, y, cv=kfold, scoring=scoring)
	save_measures(output_cross, scores)
	# y_pred = cross_val_predict(model, X, y, cv=kfold)
	# conf_mat = (pd.crosstab(y, y_pred, rownames=['REAL'], colnames=['PREDITO'], margins=True))
	# conf_mat.to_csv(matrix_output)
	return


def objective_feature_selection(space):

	"""Feature Importance-based Feature selection: Objective Function - Bayesian Optimization"""

	t = space['threshold']

	fs = SelectFromModel(clf, threshold=t)
	fs.fit(train, train_labels)
	fs_train = fs.transform(train)
	kfold = KFold(n_splits=5, shuffle=True)
	rmse = cross_val_score(clf,
						   fs_train,
						   train_labels,
						   cv=kfold,
						   scoring='neg_root_mean_squared_error',
						   n_jobs=n_cpu).mean()

	return {'loss': -rmse, 'status': STATUS_OK}


def feature_importance_fs_bayesian(model, train, train_labels):

	"""Feature Importance-based Feature selection using Bayesian Optimization"""

	model.fit(train, train_labels)

	importances = set(model.feature_importances_)
	# print(importances)
	importances.remove(max(importances))
	importances.remove(max(importances))

	space = {'threshold': hp.uniform('threshold', min(importances), max(importances))}

	trials = Trials()
	best_threshold = fmin(fn=objective_feature_selection,
					   space=space,
					   algo=tpe.suggest,
					   max_evals=100,
					   trials=trials)

	return best_threshold['threshold']


def features_importance_ensembles(model, features, output_importances):

	"""Generate feature importance values"""

	file = open(output_importances, 'a')
	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]
	names = [features[i] for i in indices]
	for f in range(len(features)):
		file.write('%d. Feature (%s): (%f)' % (f + 1, names[f], importances[indices[f]]))
		file.write('\n')
		#  print('%d. %s: (%f)' % (f + 1, names[f], importances[indices[f]]))
	return names


def save_prediction(prediction, nameseqs, pred_output):

	"""Saving prediction - test set"""

	file = open(pred_output, 'a')

	if os.path.exists(nameseq_test) is True:
		for i in range(len(prediction)):
			file.write('%s,' % str(nameseqs[i]))
			file.write('%s' % str(prediction[i]))
			file.write('\n')
	else:
		for i in range(len(prediction)):
			file.write('%s' % str(prediction[i]))
			file.write('\n')
	return


def regression_pipeline(test, test_labels, test_nameseq, fs, classifier, output):

	global clf, train, train_labels

	if not os.path.exists(output):
		os.mkdir(output)

	train = train_read
	train_labels = train_labels_read
	column_train = train.columns
	column_test = ''
	output = output + '/'

	#  tmp = sys.stdout
	#  log_file = open(output + 'task.log', 'a')
	#  sys.stdout = log_file

	"""Number of Samples and Features: Train and Test"""

	print('Number of samples (train): ' + str(len(train)))

	if os.path.exists(ftest) is True:
		column_test = test.columns
		print('Number of samples (test): ' + str(len(test)))

	print('Number of features (train): ' + str(len(column_train)))

	if os.path.exists(ftest_labels) is True:
		print('Number of features (test): ' + str(len(column_test)))

	"""Preprocessing:  Missing Values"""

	print('Checking missing values...')
	missing = train.isnull().values.any()
	inf = train.isin([np.inf, -np.inf]).values.any()
	missing_test = False
	inf_test = False
	if os.path.exists(ftest) is True:
		missing_test = test.isnull().values.any()
		inf_test = test.isin([np.inf, -np.inf]).values.any()
	if missing or inf or missing_test or inf_test:
		print('There are missing values...')
		print('Applying SimpleImputer - strategy (mean)...')
		train.replace([np.inf, -np.inf], np.nan, inplace=True)
		imp = SimpleImputer(missing_values=np.nan, strategy='mean')
		train = pd.DataFrame(imp.fit_transform(train), columns=column_train)
		if os.path.exists(ftest) is True:
			test.replace([np.inf, -np.inf], np.nan, inplace=True)
			test = pd.DataFrame(imp.transform(test), columns=column_test)
		else:
			pass
	else:
		print('There are no missing values...')

	"""Choosing Classifier """

	print('Choosing Classifier...')
	if classifier == 0:
		print('CatBoostRegressor')
		clf = CatBoostRegressor(n_estimators=200, thread_count=n_cpu,
								nan_mode='Max', logging_level='Silent', random_state=63)
	elif classifier == 1:
		print('RandomForestRegressor')
		clf = RandomForestRegressor(n_estimators=200, n_jobs=n_cpu, random_state=63)
	elif classifier == 2:
		print('XGBRegressor')
		clf = xgb.XGBRegressor(n_estimators=500, n_jobs=n_cpu, random_state=63)
	else:
		print('LGBMRegressor')
		clf = lgb.LGBMRegressor(n_estimators=500, n_jobs=n_cpu, random_state=63)

	"""Preprocessing: Feature Importance-Based Feature Selection"""

	feature_name = column_train
	if fs == 1:
		print('Applying Feature Importance-Based Feature Selection...')
		# best_t, best_baac = feature_importance_fs(clf, train, train_labels, column_train)
		best_t = feature_importance_fs_bayesian(clf, train, train_labels)
		fs = SelectFromModel(clf, threshold=best_t)
		fs.fit(train, train_labels)
		feature_idx = fs.get_support()
		feature_name = column_train[feature_idx]
		train = pd.DataFrame(fs.transform(train), columns=feature_name)
		if os.path.exists(ftest) is True:
			test = pd.DataFrame(fs.transform(test), columns=feature_name)
		else:
			pass
		print('Best Feature Subset: ' + str(len(feature_name)))
		print('Reduction: ' + str(len(column_train)-len(feature_name)) + ' features')
		fs_train = output + 'best_feature_train.csv'
		fs_test = output + 'best_feature_test.csv'
		print('Saving dataset with selected feature subset - train: ' + fs_train)
		train.to_csv(fs_train, index=False)
		if os.path.exists(ftest) is True:
			print('Saving dataset with selected feature subset - test: ' + fs_test)
			test.to_csv(fs_test, index=False)
		print('Feature Selection - Finished...')

	"""Training - StratifiedKFold (cross-validation = 10)..."""

	print('Training: StratifiedKFold (cross-validation = 10)...')
	train_output = output + 'training_kfold(10)_metrics.csv'
	model_output = output + 'trained_model.sav'
	evaluate_model_cross(train, train_labels, clf, train_output)
	clf.fit(train, train_labels)
	joblib.dump(clf, model_output)
	print('Saving results in ' + train_output + '...')
	print('Saving trained model in ' + model_output + '...')
	print('Training: Finished...')

	"""Generating Feature Importance - Selected feature subset..."""

	print('Generating Feature Importance - Selected feature subset...')
	importance_output = output + 'feature_importance.csv'
	features_importance_ensembles(clf, feature_name, importance_output)
	print('Saving results in ' + importance_output + '...')

	"""Testing model..."""

	if os.path.exists(ftest) is True:
		print('Generating Performance Test...')
		preds = clf.predict(test)
		pred_output = output + 'test_predictions.csv'
		print('Saving prediction in ' + pred_output + '...')
		save_prediction(preds, test_nameseq, pred_output)
		if os.path.exists(ftest_labels) is True:
			print('Generating Metrics - Test set...')
			MAE = mean_absolute_error(test_labels, preds)
			MSE = mean_squared_error(test_labels, preds)
			RMSE = mean_squared_error(test_labels, preds, squared=False)
			MAE_median = median_absolute_error(test_labels, preds)
			R2 = r2_score(test_labels, preds)
			metrics_output = output + 'metrics_test.csv'
			print('Saving Metrics - Test set: ' + metrics_output + '...')
			file = open(metrics_output, 'a')
			file.write('Metrics: Test Set')
			file.write('\n')
			file.write('MAE: %s' % MAE)
			file.write('\n')
			file.write('MSE: %s' % MSE)
			file.write('\n')
			file.write('RMSE: %s' % RMSE)
			file.write('\n')
			file.write('MAE_median: %s' % MAE_median)
			file.write('\n')
			file.write('R2: %s' % R2)
			print('Task completed - results generated in ' + output + '!')

		else:
			print('There are no test labels for evaluation, check parameters...')
			#  sys.stdout = tmp
			#  log_file.close()
	else:
		print('There are no test sequences for evaluation, check parameters...')
		print('Task completed - results generated in ' + output + '!')
		#  sys.stdout = tmp
		#  log_file.close()

	return


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
##                                      Regression module                                         ##
##                                                                                                ##
####################################################################################################
####################################################################################################
    ''')
	parser = argparse.ArgumentParser()
	parser.add_argument('-train', '--train', help='csv format file, e.g., train.csv')
	parser.add_argument('-train_label', '--train_label', default='', help='csv format file, e.g., labels.csv')
	parser.add_argument('-test', '--test', help='csv format file, e.g., train.csv')
	parser.add_argument('-test_label', '--test_label', default='', help='csv format file, e.g., labels.csv')
	parser.add_argument('-test_nameseq', '--test_nameseq', default='', help='csv with sequence names')
	parser.add_argument('-fs', '--featureselection', default=1,
						help='Feature Selection (default = True)')
	parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
	parser.add_argument('-classifier', '--classifier', default=0,
						help='Classifier - 0: CatBoost, 1: Random Forest'
							 '2: XGBoost, 3: LightGBM')
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')
	args = parser.parse_args()
	ftrain = str(args.train)
	ftrain_labels = str(args.train_label)
	ftest = str(args.test)
	ftest_labels = str(args.test_label)
	nameseq_test = str(args.test_nameseq)
	fs = int(args.featureselection)
	n_cpu = int(args.n_cpu)
	classifier = int(args.classifier)
	foutput = str(args.output)
	start_time = time.time()

	if os.path.exists(ftrain) is True:
		train_read = pd.read_csv(ftrain)
		print('Train - %s: Found File' % ftrain)
	else:
		print('Train - %s: File not exists' % ftrain)
		sys.exit()

	if os.path.exists(ftrain_labels) is True:
		train_labels_read = pd.read_csv(ftrain_labels).values.ravel()
		print('Train_labels - %s: Found File' % ftrain_labels)
	else:
		print('Train_labels - %s: File not exists' % ftrain_labels)
		sys.exit()

	test_read = ''
	if ftest != '':
		if os.path.exists(ftest) is True:
			test_read = pd.read_csv(ftest)
			print('Test - %s: Found File' % ftest)
		else:
			print('Test - %s: File not exists' % ftest)
			sys.exit()

	test_labels_read = ''
	if ftest_labels != '':
		if os.path.exists(ftest_labels) is True:
			test_labels_read = pd.read_csv(ftest_labels).values.ravel()
			print('Test_labels - %s: Found File' % ftest_labels)
		else:
			print('Test_labels - %s: File not exists' % ftest_labels)
			sys.exit()

	test_nameseq_read = ''
	if nameseq_test != '':
		if os.path.exists(nameseq_test) is True:
			test_nameseq_read = pd.read_csv(nameseq_test).values.ravel()
			print('Test_nameseq - %s: Found File' % nameseq_test)
		else:
			print('Test_nameseq - %s: File not exists' % nameseq_test)
			sys.exit()

	regression_pipeline(test_read, test_labels_read, test_nameseq_read,
						fs, classifier, foutput)
	cost = (time.time() - start_time)/60
	print('Computation time - Pipeline: %s minutes' % cost)
##########################################################################
##########################################################################
