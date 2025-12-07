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
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from tpot import TPOTClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score

def header(output_header):

	"""Header Function: Header of the evaluate_model_cross Function"""

	file = open(output_header, 'a')
	file.write('mean_absolute_error,std_mean_absolute_error,mean_squared_error'
			   ',std_mean_squared_error,root_mean_squared_error,std_root_mean_squared_error,'
			   'r2,std_r2')
	file.write('\n')
	return


def save_measures(output_measures, scores):

	"""Save Measures Function: Output of the evaluate_model_cross Function"""

	header(output_measures)
	file = open(output_measures, 'a')
	file.write('%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f' 
            	% (
					abs(scores['test_MAE']).mean(),
					+ abs(scores['test_MAE']).std(), abs(scores['test_MSE']).mean(),
					+ abs(scores['test_MSE']).std(),
					+ abs(scores['test_RMSE']).mean(), abs(scores['test_RMSE']).std(),
					+ scores['test_R2'].mean(), scores['test_R2'].std())
                )
	file.write('\n')
	return


def evaluate_model_cross(X, y, model, output_cross):

	"""Evaluation Function: Using Cross-Validation"""

	scoring = {'MAE': 'neg_mean_absolute_error',
			   'MSE': 'neg_mean_squared_error',
			   'RMSE': 'neg_root_mean_squared_error',
			   'R2': 'r2'}
	kfold = KFold(n_splits=5, shuffle=True)
	scores = cross_validate(model, X, y, cv=kfold, scoring=scoring)
	save_measures(output_cross, scores)

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

	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]
	names = [features[i] for i in indices]
	with open(output_importances, 'w') as file:
		file.write('Feature\tImportance\n')  # Write header
		for f in range(len(features)):
			file.write('%s\t%f\n' % (names[f], importances[indices[f]]))

	return names


def save_prediction(prediction, nameseqs, pred_output):

	"""Saving prediction - test set"""

	preds_df = pd.DataFrame({"nameseq": nameseqs, "prediction": prediction})

	preds_df.to_csv(pred_output, index=False)

def regression_pipeline(model, train, train_labels, train_nameseq, test, test_labels, test_nameseq, norm, classifier, fs, output):
	"""Unified Regression Pipeline — mirrors the structure of binary_pipeline"""
	global clf, ord_encoder

	if not os.path.exists(output):
		os.mkdir(output)

	# Initialize model dictionary
	if model:
		train = model["train"]
		train_labels = model["train_labels"]
		column_train = model["column_train"]
	else:
		column_train = train.columns
		model_dict = {"train": train, "train_labels": train_labels, "column_train": column_train}

	column_test = ''

	"""Basic Info"""
	print(f'Number of samples (train): {len(train)}')
	print(f'Number of features (train): {len(column_train)}')

	if os.path.exists(ftest):
		column_test = test.columns
		print(f'Number of samples (test): {len(test)}')
		print(f'Number of features (test): {len(column_test)}')

	"""Preprocessing: Ordinal Encoding (categorical features)"""
	if model:
		if "ordinal_encoder" in model:
			ord_encoder = model["ordinal_encoder"]
			string_cols = train.select_dtypes(include=["object"]).columns
			if not string_cols.empty:
				train[string_cols] = ord_encoder.transform(train[string_cols])
	else:
		string_cols = train.select_dtypes(include=["object"]).columns
		if not string_cols.empty:
			print('Applying OrdinalEncoder()...')
			ord_encoder = OrdinalEncoder()
			train[string_cols] = ord_encoder.fit_transform(train[string_cols])
			if os.path.exists(ftest):
				string_cols = test.select_dtypes(include=["object"]).columns
				if not string_cols.empty:
					test[string_cols] = ord_encoder.transform(test[string_cols])
			model_dict["ordinal_encoder"] = ord_encoder

	"""Preprocessing: Missing Values"""
	print('Checking missing values...')
	missing = train.isnull().values.any()
	inf = train.isin([np.inf, -np.inf]).values.any()
	missing_test = inf_test = False
	if os.path.exists(ftest):
		missing_test = test.isnull().values.any()
		inf_test = test.isin([np.inf, -np.inf]).values.any()
	if missing or inf or missing_test or inf_test:
		print('There are missing or infinite values — applying SimpleImputer(mean)')
		train.replace([np.inf, -np.inf], np.nan, inplace=True)
		imp = SimpleImputer(strategy='mean')
		train = pd.DataFrame(imp.fit_transform(train), columns=column_train)
		if os.path.exists(ftest):
			test.replace([np.inf, -np.inf], np.nan, inplace=True)
			test = pd.DataFrame(imp.transform(test), columns=column_test)
		model_dict["imputer"] = imp
	else:
		print('No missing values found.')

	"""Preprocessing: Normalization"""
	if norm:
		print('Applying StandardScaler()...')
		if model:
			sc = model["scaler"]
			train = pd.DataFrame(sc.transform(train), columns=column_train)
		else:
			sc = StandardScaler()
			train = pd.DataFrame(sc.fit_transform(train), columns=column_train)
			model_dict["scaler"] = sc
		if os.path.exists(ftest):
			test = pd.DataFrame(sc.transform(test), columns=column_test)

	"""Choosing Regressor"""
	if not model:
		if classifier == 0:
			print('Regressor: CatBoostRegressor')
			clf = CatBoostRegressor(n_estimators=500, thread_count=n_cpu, nan_mode='Max',
									logging_level='Silent', random_state=63)
		elif classifier == 1 or classifier == 2 or classifier == 3:
			print('Regressor: LightGBM')
			clf = lgb.LGBMRegressor(n_estimators=500, n_jobs=n_cpu, random_state=63, verbosity=-1)
	else:
		clf = model["clf"]

	"""Feature Selection"""
	feature_name = column_train
	if not model:
		if fs:
			print('Applying Feature Importance-Based Feature Selection...')
			best_t = feature_importance_fs_bayesian(clf, train, train_labels)
			selector = SelectFromModel(clf, threshold=best_t)
			selector.fit(train, train_labels)
			feature_idx = selector.get_support()
			feature_name = column_train[feature_idx]
			train = pd.DataFrame(selector.transform(train), columns=feature_name)
			if os.path.exists(ftest):
				test = pd.DataFrame(selector.transform(test), columns=feature_name)
			print(f'Feature Selection Done — Retained: {len(feature_name)} / {len(column_train)} features')
			model_dict["fs"] = selector
	else:
		if "fs" in model:
			feature_idx = model["fs"].get_support()
			feature_name = column_train[feature_idx]
			train = pd.DataFrame(model["fs"].transform(train), columns=feature_name)
			if os.path.exists(ftest):
				test = pd.DataFrame(model["fs"].transform(test), columns=feature_name)

	"""Training and Cross-Validation"""
	print('Training: KFold (cross-validation = 10)...')
	train_output = os.path.join(output, 'training_kfold(10)_metrics.csv')
	importance_output = os.path.join(output, 'feature_importance.csv')
	model_output = os.path.join(output, 'trained_model.sav')

	if not model:
		evaluate_model_cross(train, train_labels, clf, train_output)
		clf.fit(train, train_labels)
		model_dict["clf"] = clf
		print(f'Saving training metrics → {train_output}')
		print(f'Saving trained model → {model_output}')

		features_importance_ensembles(clf, feature_name, importance_output)
		print(f'Saving feature importance → {importance_output}')

		model_dict["feature_importance"] = pd.read_csv(importance_output, sep='\t')
		model_dict["nameseq_train"] = train_nameseq
		joblib.dump(model_dict, model_output)
	else:
		clf = model["clf"]

	"""Testing"""
	if os.path.exists(ftest):
		print('Generating Performance on Test Set...')
		preds = clf.predict(test)
		pred_output = os.path.join(output, 'test_predictions.csv')
		save_prediction(preds, test_nameseq, pred_output)
		print(f'Saving predictions → {pred_output}')

		if os.path.exists(ftest_labels):
			MAE = mean_absolute_error(test_labels, preds)
			MSE = mean_squared_error(test_labels, preds)
			RMSE = root_mean_squared_error(test_labels, preds)
			R2 = r2_score(test_labels, preds)
			metrics = pd.DataFrame({
				"Metric": ["MAE", "MSE", "RMSE", "R2"],
				"Value": [MAE, MSE, RMSE, R2]
			})
			metrics_output = os.path.join(output, 'metrics_test.csv')
			metrics.to_csv(metrics_output, index=False)
			print(f'Saving test metrics → {metrics_output}')
			print('Task completed successfully!')
		else:
			print('No test labels provided for evaluation.')
	else:
		print('No test data found — skipping testing.')

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
	parser.add_argument('-path_model', '--path_model', default='', help='Path to trained model to be used.')
	parser.add_argument('-train', '--train', help='csv format file, e.g., train.csv')
	parser.add_argument('-train_label', '--train_label', default='', help='csv format file, e.g., labels.csv')
	parser.add_argument('-train_nameseq', '--train_nameseq', default='', help='csv with sequence names')
	parser.add_argument('-test', '--test', help='csv format file, e.g., train.csv')
	parser.add_argument('-test_label', '--test_label', default='', help='csv format file, e.g., labels.csv')
	parser.add_argument('-test_nameseq', '--test_nameseq', default='', help='csv with sequence names')
	parser.add_argument('-nf', '--normalization', type=bool, default=False, help='Normalization - Features (default = False)')
	parser.add_argument('-fselection', '--fselection', default=0,
						help='Feature Selection (default = True)')
	parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
	parser.add_argument('-classifier', '--classifier', default=1,
						help='Classifier - 0: CatBoost, 1: LightGBM')
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')
	args = parser.parse_args()
	path_model = args.path_model
	ftrain = str(args.train)
	ftrain_labels = str(args.train_label)
	nameseq_train = str(args.train_nameseq)
	ftest = str(args.test)
	ftest_labels = str(args.test_label)
	nameseq_test = str(args.test_nameseq)
	norm = args.normalization
	fs = int(args.fselection)
	n_cpu = int(args.n_cpu)
	classifier = int(args.classifier)
	foutput = str(args.output)
	start_time = time.time()

	model = ''
	train_read, train_labels_read, train_nameseq_read = '', '', ''
	if path_model:
		model = joblib.load(path_model)
	else:
		if os.path.exists(ftrain) is True:
			train_read = pd.read_csv(ftrain)
			print('Train - %s: Found File' % ftrain)
		else:
			print('Train - %s: File not exists' % ftrain)
			sys.exit()

		if os.path.exists(nameseq_train) is True:
			train_nameseq_read = pd.read_csv(nameseq_train).values.ravel()
			print('Train_nameseq - %s: Found File' % nameseq_train)
		else:
			print('Train_nameseq - %s: File not exists' % nameseq_train)
			sys.exit()
                  
		if os.path.exists(ftrain_labels) is True:
			train_labels_read = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(nameseq_train)["nameseq"].to_list()]
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

	test_nameseq_read = ''
	if nameseq_test != '':
		if os.path.exists(nameseq_test) is True:
			test_nameseq_read = pd.read_csv(nameseq_test).values.ravel()
			print('Test_nameseq - %s: Found File' % nameseq_test)
		else:
			print('Test_nameseq - %s: File not exists' % nameseq_test)
			sys.exit()

	test_labels_read = ''
	if ftest_labels != '':
		if os.path.exists(ftest_labels) is True:
			test_labels_read = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(nameseq_test)["nameseq"].to_list()]
			print('Test_labels - %s: Found File' % ftest_labels)
		else:
			print('Test_labels - %s: File not exists' % ftest_labels)
			sys.exit()

	regression_pipeline(model, train_read, train_labels_read, train_nameseq_read, 
					 	test_read, test_labels_read, test_nameseq_read,
						norm, classifier, fs, foutput)
	cost = (time.time() - start_time)/60
	print('Computation time - Pipeline: %s minutes' % cost)
##########################################################################
##########################################################################
