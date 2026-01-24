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
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef, classification_report
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import make_scorer, matthews_corrcoef, cohen_kappa_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from tpot import TPOTClassifier
from numpy.random import default_rng
from functools import partial
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score

def save_measures(output_measures, scores):
    """
    Save cross-validation measures for classification or regression.

    This function automatically adapts to the metrics present in `scores`
    (binary classification, multiclass classification, or regression).
    """

    # Define preferred order for known metrics
    preferred_metrics = [
        "ACC", "Sn", "Sp", "F1", "F1_macro", "F1_micro", "F1_weighted",
        "MCC", "AUC", "ACC_B", "kappa", "gmean",
        "MAE", "MSE", "RMSE", "R2"
    ]

    results = {}
    available_metrics = []

    # Detect available test metrics
    for key in scores.keys():
        if key.startswith("test_"):
            metric = key.replace("test_", "")
            available_metrics.append(metric)

    # Sort metrics: preferred order first, then any extras
    ordered_metrics = (
        [m for m in preferred_metrics if m in available_metrics] +
        sorted(set(available_metrics) - set(preferred_metrics))
    )

    # Compute mean and std safely
    for metric in ordered_metrics:
        values = scores.get(f"test_{metric}")
        if values is None:
            continue

        mean_val = np.mean(values)
        std_val = np.std(values)

        # Convert negative regression losses to positive values
        if metric in {"MAE", "MSE", "RMSE"}:
            mean_val = abs(mean_val)

        results[metric] = round(mean_val, 4)
        results[f"std_{metric}"] = round(std_val, 4)

    # Build DataFrame (single-row)
    df = pd.DataFrame([results])

    # Write to CSV
    df.to_csv(
        output_measures,
        index=False,
    )

def evaluate_model_cross(X, y, model, task, output_cross, matrix_output):
    """Evaluation Function: Using Cross-Validation"""

    def specificity_score(y_true, y_pred):
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def specificity_score_macro(y_true, y_pred):
        labels = np.unique(y_true)
        specs = []
        for label in labels:
            tn = ((y_true != label) & (y_pred != label)).sum()
            fp = ((y_true != label) & (y_pred == label)).sum()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specs.append(spec)
        return np.mean(specs)

    if task == 0:
        if len(np.unique(y)) > 2:
            scoring = {
                'ACC': make_scorer(accuracy_score),
                'Sn': make_scorer(recall_score, average='macro'),
                'Sp': make_scorer(specificity_score_macro),
                'F1_macro': make_scorer(f1_score, average='macro'),
                'MCC': make_scorer(matthews_corrcoef),
                'kappa': make_scorer(cohen_kappa_score),
                'F1_micro': make_scorer(f1_score, average='micro'),
                'F1_weighted': make_scorer(f1_score, average='weighted')
            }
        else:
            scoring = {
                'ACC': 'accuracy',
                'Sn': make_scorer(recall_score),
                'Sp': make_scorer(specificity_score),
                'F1': make_scorer(f1_score),
                'MCC': make_scorer(matthews_corrcoef),
                'AUC': 'roc_auc',
                'ACC_B': 'balanced_accuracy',
                'kappa': make_scorer(cohen_kappa_score),
                'gmean': make_scorer(geometric_mean_score)
            }

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=63)
        scores = cross_validate(model, X, y, cv=kfold, scoring=scoring)

        save_measures(output_cross, scores)

        y_pred = cross_val_predict(model, X, y, cv=kfold)
        conf_mat = pd.crosstab(
            lb_encoder.inverse_transform(y),
            lb_encoder.inverse_transform(y_pred),
            rownames=['REAL'], colnames=['PREDICTED'], margins=True
        )

        conf_mat.to_csv(matrix_output)
    else:
        scoring = {'MAE': 'neg_mean_absolute_error',
            'MSE': 'neg_mean_squared_error',
            'RMSE': 'neg_root_mean_squared_error',
            'R2': 'r2'}

        kfold = KFold(n_splits=10, shuffle=True, random_state=63)
        scores = cross_validate(model, X, y, cv=kfold, scoring=scoring)

        save_measures(output_cross, scores)

def features_importance_ensembles(model, features, output_importances):
    """
    Generate and save feature importance values using pandas.

    Parameters
    ----------
    model : fitted model
        Must expose `feature_importances_`
    features : list of str
        Feature names
    output_importances : str
        Output file path

    Returns
    -------
    list
        Feature names sorted by descending importance
    """

    importances = model.named_steps["clf"].feature_importances_
    indices = np.argsort(importances)[::-1]

    df = pd.DataFrame({
        "Feature": [features[i] for i in indices],
        "Importance": importances[indices]
    })

    df.to_csv(
        output_importances,
        sep="\t",
        index=False,
        float_format="%.6f"
    )

    return df["Feature"].tolist()
    
def save_prediction(task, prediction, nameseqs, pred_output):
    
    """Saving prediction - test set"""

    if task == 0:
        nameseq_df = pd.DataFrame(nameseqs, columns=["nameseq"])

        probs_df = pd.DataFrame(prediction, columns=lb_encoder.classes_)
        probs_df["prediction"] = probs_df.idxmax(axis=1)

        preds_df = pd.concat([nameseq_df, probs_df], axis=1)
    else:
        preds_df = pd.DataFrame({"nameseq": nameseqs, "prediction": prediction})

    preds_df.to_csv(pred_output, index=False)

def get_best_model_optuna(X, y, task, classifier_type, n_trials):
    """
    Runs Optuna optimization and returns the best configured pipeline.
    task: 0 = Classification, 1 = Regression
    classifier_type: 0 = RF, 1 = XGB, 2 = LGBM 
    """
    
    def objective(trial):
        # Define base pipeline components
        imputer = SimpleImputer(strategy='mean')
        
        # --- CLASSIFICATION (Task 0) ---
        if task == 0:
            if classifier_type == 0: # Random Forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': 63
                }
                model = RandomForestClassifier(**params)
                
            elif classifier_type == 1: # XGBoost
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'random_state': 63,
                    'eval_metric': 'mlogloss' if len(np.unique(y)) > 2 else 'logloss'
                }
                model = xgb.XGBClassifier(**params)
                
            elif classifier_type == 2: # LightGBM
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'random_state': 63,
                    'verbosity': -1
                }
                model = lgb.LGBMClassifier(**params)

            clf_pipeline = Pipeline(steps=[("imputer", imputer), ("clf", model)])
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=63)
            score = make_scorer(matthews_corrcoef)
            scores = cross_val_score(clf_pipeline, X, y, cv=cv, scoring=score)
            return scores.mean()

        # --- REGRESSION (Task 1) ---
        elif task == 1:
            common_lgb_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'random_state': 63,
                'verbosity': -1,
                'n_jobs': 1
            }

            if classifier_type == 0: # LightGBM (RF Mode)
                # Specific overrides for RF mode
                params = common_lgb_params.copy()
                params.update({
                    'boosting_type': 'rf',
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9)
                })
                model = lgb.LGBMRegressor(**params)

            elif classifier_type == 1: # LightGBM (Random Hist / GBDT)
                params = common_lgb_params.copy()
                params.update({
                    'boosting_type': 'gbdt',
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50)
                })
                model = lgb.LGBMRegressor(**params)
                
            elif classifier_type == 2: # LightGBM (Standard)
                params = common_lgb_params.copy()
                model = lgb.LGBMRegressor(**params)

            # Metric: Negative RMSE for regression (Optuna maximizes return value)
            reg_pipeline = Pipeline(steps=[("imputer", imputer), ("clf", model)])
            cv = KFold(n_splits=10, shuffle=True, random_state=63)
            score = make_scorer(root_mean_squared_error)
            scores = cross_val_score(reg_pipeline, X, y, cv=cv, scoring=score)
            return scores.mean()

    # Create Study
    if task == 0:
        direction = "maximize"
    else:
        direction = "minimize"

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True))
    study.optimize(objective, n_trials=n_trials, timeout=7200, show_progress_bar=True, n_jobs=32)
    
    print(f"Best Trial: {study.best_value:.4f}")
    print("Best Params: ", study.best_params)

    # Reconstruct the best model pipeline
    best_params = study.best_params
    imputer = SimpleImputer(strategy='mean')
    
    # Re-instantiate based on task/type with best params
    if task == 0:
        if classifier_type == 0:
            final_model = RandomForestClassifier(random_state=63, **best_params)
        elif classifier_type == 1:
            # Handle xgboost eval_metric outside params if needed or ensure it's in best_params handling
            metric = 'mlogloss' if len(np.unique(y)) > 2 else 'logloss'
            final_model = xgb.XGBClassifier(random_state=63, eval_metric=metric, **best_params)
        elif classifier_type == 2:
            final_model = lgb.LGBMClassifier(random_state=63, verbosity=-1, **best_params)
    else:
        # For regression, ensure static params (like boosting_type) are re-added if not optimized
        if classifier_type == 0:
            final_model = lgb.LGBMRegressor(boosting_type='rf', random_state=63, verbosity=-1, n_jobs=1, **best_params)
        elif classifier_type == 1:
            final_model = lgb.LGBMRegressor(boosting_type='gbdt', random_state=63, verbosity=-1, n_jobs=1, **best_params)
        elif classifier_type == 2:
            final_model = lgb.LGBMRegressor(random_state=63, verbosity=-1, n_jobs=1, **best_params)

    return Pipeline(steps=[("imputer", imputer), ("clf", final_model)])

def predictive_pipeline(model, task, tuning, train, train_labels, train_nameseq, test, test_labels, test_nameseq, classifier, output):
    
    global clf, lb_encoder, ord_encoder

    if not os.path.exists(output):
        os.mkdir(output)

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

    """Preprocessing:  Label Encoding"""

    if model:
        lb_encoder = model["label_encoder"]
        ord_encoder = model["ordinal_encoder"]

        if task == 0:
            train_labels = lb_encoder.transform(train_labels)

        string_cols = train.select_dtypes(include=["object"]).columns
        if not string_cols.empty:
            train[string_cols] = ord_encoder.transform(train[string_cols])
    else:
        lb_encoder, ord_encoder = LabelEncoder(), OrdinalEncoder()

        if task == 0:
            train_labels = lb_encoder.fit_transform(train_labels)

        string_cols = train.select_dtypes(include=["object"]).columns
        if not string_cols.empty:
            train[string_cols] = ord_encoder.fit_transform(train[string_cols])

        if os.path.exists(ftest) is True:
            string_cols = test.select_dtypes(include=["object"]).columns
            if not string_cols.empty:
                test[string_cols] = ord_encoder.transform(test[string_cols])

        if task == 0:
            model_dict["label_encoder"] = lb_encoder
        model_dict["ordinal_encoder"] = ord_encoder
    
    """Preprocessing:  Missing Values"""

    print('Checking missing values...')

    if model:
        if "imputer" in model:
            imp = model["imputer"]
            print('Applying SimpleImputer - strategy (mean)...')

            if os.path.exists(ftest):
                test = test.replace([np.inf, -np.inf], np.nan)
                test = pd.DataFrame(imp.transform(test), columns=column_test)
    else:
        print('Applying SimpleImputer - strategy (mean)...')
        
        imp = SimpleImputer(strategy='mean')

        train = train.replace([np.inf, -np.inf], np.nan)
        model_dict["imputer"] = imp.fit(train)

        if os.path.exists(ftest) is True:
            test = test.replace([np.inf, -np.inf], np.nan)
            test = pd.DataFrame(imp.transform(test), columns=column_test)

    """Choosing Classifier """

    if not model:
        sc = StandardScaler()
        model_dict["scaler"] = sc.fit(train)

        print('--- Optimizing Hyperparameters with Optuna ---')
        # We replace the hardcoded logic with the optimization function call
        clf = get_best_model_optuna(train, train_labels, task, classifier, tuning)
        print('--- Optimization Complete ---')
        
        # Mapping names for logging purposes
        if task == 0:
            names = ["Random Forest", "XGBoost", "LightGBM"]
            print(f"Selected Classifier: {names[classifier]} (Optimized)")
        else:
            names = ["LightGBM (RF)", "LightGBM (Random Hist)", "LightGBM"]
            print(f"Selected Regressor: {names[classifier]} (Optimized)")
        # if task == 0:
        #     if classifier == 0:
        #         print('Classifier: Random Forest')

        #         clf = Pipeline(steps=[
        #             ("imputer", SimpleImputer(strategy="mean")),
        #             ("clf", RandomForestClassifier(
        #                 n_estimators=200,
        #                 random_state=63,
        #             ))
        #         ])

        #     elif classifier == 1:
        #         print('Classifier: XGBoost')

        #         if len(np.unique(train_labels)) > 2:
        #             clf = Pipeline(steps=[
        #                 ("imputer", SimpleImputer(strategy="mean")),
        #                 ("clf", xgb.XGBClassifier(
        #                     eval_metric="logloss",
        #                     random_state=63,
        #                 ))
        #             ])
        #         else:
        #             clf = Pipeline(steps=[
        #                 ("imputer", SimpleImputer(strategy="mean")),
        #                 ("clf", xgb.XGBClassifier(
        #                     eval_metric="mlogloss",
        #                     random_state=63,
        #                 ))
        #             ])  
        #     elif classifier == 2:
        #         print('Classifier: LightGBM')

        #         clf = Pipeline(steps=[
        #             ("imputer", SimpleImputer(strategy="mean")),
        #             ("clf", lgb.LGBMClassifier(
        #                 n_estimators=500,
        #                 random_state=63,
        #                 verbosity=-1
        #             ))
        #         ])
        # elif task == 1:
        #     if classifier == 0:
        #         print('Regressor: LightGBM (RF)')

        #         clf = Pipeline(steps=[
        #             ("imputer", SimpleImputer(strategy="mean")),
        #             ("clf", lgb.LGBMRegressor(
        #                 boosting_type='rf',
        #                 n_estimators=500,
        #                 bagging_freq=1,
        #                 bagging_fraction=0.8,
        #                 feature_fraction=0.8,
        #                 random_state=63,
        #                 verbosity=-1,
        #                 n_jobs=1
        #             ))
        #         ])

        #     elif classifier == 1:
        #         print('Regressor: LightGBM (Random Hist)')

        #         clf = Pipeline(steps=[
        #             ("imputer", SimpleImputer(strategy="mean")),
        #             ("clf", lgb.LGBMRegressor(
        #                 boosting_type='gbdt',
        #                 n_estimators=500,
        #                 feature_fraction=0.7,
        #                 bagging_fraction=0.7,
        #                 bagging_freq=1,
        #                 min_data_in_leaf=20,
        #                 random_state=63,
        #                 verbosity=-1,
        #                 n_jobs=1
        #             ))
        #         ])
        #     elif classifier == 2:
        #         print('Regressor: LightGBM')

        #         clf = Pipeline(steps=[
        #             ("imputer", SimpleImputer(strategy="mean")),
        #             ("clf", lgb.LGBMRegressor(n_estimators=500, n_jobs=1, random_state=63, verbosity=-1))
        #         ])

    """Training - StratifiedKFold (cross-validation = 10)..."""

    print('Training: StratifiedKFold (cross-validation = 10)...')
    
    train_output = os.path.join(output, 'training_kfold(10)_metrics.csv')
    matrix_output = os.path.join(output, 'training_confusion_matrix.csv')
    importance_output = os.path.join(output, 'feature_importance.tsv')
    descriptors_output = os.path.join(output, 'best_descriptors/selected_descriptors.csv')
    model_output = os.path.join(output, 'trained_model.sav')

    if model:
        clf = model["clf"]
    else:
        evaluate_model_cross(train, train_labels, clf, task, train_output, matrix_output)

        clf.fit(train, train_labels)
        model_dict["clf"] = clf

        model_dict["cross_validation"] = pd.read_csv(train_output)

        if task == 0:
            model_dict["confusion_matrix"] = pd.read_csv(matrix_output)

        if os.path.exists(descriptors_output):    
            model_dict["descriptors"] = pd.read_csv(descriptors_output)
        model_dict["nameseq_train"] = train_nameseq
        
        print('Saving results in ' + train_output + '...')
        print('Saving confusion matrix in ' + matrix_output + '...')
        print('Saving trained model in ' + model_output + '...')
        print('Training: Finished...')

        """Generating Feature Importance - Selected feature subset..."""

        print('Generating Feature Importance - Selected feature subset...')
        features_importance_ensembles(clf, column_train, importance_output)
        print('Saving results in ' + importance_output + '...')

        model_dict["feature_importance"] = pd.read_csv(importance_output, sep='\t')

        joblib.dump(model_dict, model_output)

    """Testing model..."""

    if os.path.exists(ftest) is True:
        print('Generating Performance Test...')

        if task == 0:
            preds = lb_encoder.inverse_transform(clf.predict(test))
            probs = clf.predict_proba(test)
            pred_output = os.path.join(output, "test_predictions.csv")
            print('Saving prediction in ' + pred_output + '...')
            save_prediction(task, probs, test_nameseq, pred_output)
        else:
            preds = clf.predict(test)
            pred_output = os.path.join(output, 'test_predictions.csv')
            save_prediction(task, preds, test_nameseq, pred_output)

        if os.path.exists(ftest_labels) is True and len(np.unique(test_labels)) > 1:
            print('Generating Metrics - Test set...')
            
            if task == 0:
                report = classification_report(test_labels, preds, output_dict=True)

                metrics_output = os.path.join(output, "metrics_test.csv")
                print('Saving Metrics - Test set: ' + metrics_output + '...')
                
                metr_report = pd.DataFrame(report).transpose()
                metr_report.to_csv(metrics_output)
                
                if not len(np.unique(train_labels)) > 2:
                    metrics_other_output = os.path.join(output, "metrics_other.csv")
                    accu = accuracy_score(test_labels, preds)
                    auc = roc_auc_score(test_labels, clf.predict_proba(test)[:, 1])
                    balanced = balanced_accuracy_score(test_labels, preds)
                    gmean = geometric_mean_score(test_labels, preds)
                    mcc = matthews_corrcoef(test_labels, preds)
                    
                    metrics = {
                        'Metric': ['Accuracy', 'AUC', 'Balanced ACC', 'G-mean', 'MCC'],
                        'Value': [accu, auc, balanced, gmean, mcc]
                    }

                    metrics_df = pd.DataFrame(metrics)
                    metrics_df.to_csv(metrics_other_output, index=False)

                matrix_test = (pd.crosstab(test_labels, preds, rownames=["REAL"], colnames=["PREDICTED"], margins=True))
                matrix_output_test = os.path.join(output, "test_confusion_matrix.csv")
                matrix_test.to_csv(matrix_output_test)
                print('Saving confusion matrix in ' + matrix_output_test + '...')
                print('Task completed - results generated in ' + output + '!')
            elif task == 1:
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
            print('There are no test labels for evaluation, check parameters...')
    else:
        print('There are no test sequences for evaluation, check parameters...')
        print('Task completed - results generated in ' + output + '!')

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
##           Empowering Breakthroughs in Life Sciences with End-to-End Machine Learning           ##
##                                                                                                ##
##                                    Generation module                                           ##
##                                                                                                ##
####################################################################################################
####################################################################################################
    ''')
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_model', '--path_model', default='', help='Path to trained model to be used.')
    parser.add_argument('-task', '--task', default=0, help='Machine learning task - 0: Classification, 1: Regression - Default: Classification')
    parser.add_argument('-tuning', '--tuning', default=50, help='number of trials for hyperparameter tuning - default = 50')
    parser.add_argument('-train', '--train', help='csv format file, e.g., train.csv')
    parser.add_argument('-train_label', '--train_label', default='', help='csv format file, e.g., labels.csv')
    parser.add_argument('-train_nameseq', '--train_nameseq', default='', help='csv with sequence names')
    parser.add_argument('-test', '--test', default='', help='csv format file, e.g., test.csv')
    parser.add_argument('-test_label', '--test_label', default='', help='csv format file, e.g., labels.csv')
    parser.add_argument('-test_nameseq', '--test_nameseq', default='', help='csv with sequence names')
    parser.add_argument('-n_cpu', '--n_cpu', default=-1, help='number of cpus - default = 1')
    parser.add_argument('-classifier', '--classifier', default=0,
                        help='Classifier - 0: Random Forest, 1: Random Forest, 2: XGBoost, 3: LightGBM')
    parser.add_argument('-output', '--output', help='results directory, e.g., result/')
    args = parser.parse_args()
    path_model = args.path_model
    task = int(args.task)
    tuning = int(args.tuning)
    ftrain = str(args.train)
    ftrain_labels = str(args.train_label)
    nameseq_train = str(args.train_nameseq)
    ftest = str(args.test)
    ftest_labels = str(args.test_label)
    nameseq_test = str(args.test_nameseq)
    n_cpu = int(args.n_cpu)
    classifier = int(args.classifier)
    foutput = str(args.output)
    start_time = time.time()

    model = ''
    train_read, train_labels_read, train_nameseq_read = '', '', ''
    if path_model:
        model = joblib.load(path_model)
    else:
        if os.path.exists(ftrain):
            train_read = pd.read_csv(ftrain)
            print('Train - %s: Found File' % ftrain)
        else:
            print('Train - %s: File not exists' % ftrain)
            sys.exit()

        if os.path.exists(nameseq_train):
            train_nameseq_read = pd.read_csv(nameseq_train).values.ravel()
            print('Train_nameseq - %s: Found File' % nameseq_train)
        else:
            print('Train_nameseq - %s: File not exists' % nameseq_train)
            sys.exit()

        if task == 0:
            if os.path.exists(ftrain_labels):
                train_labels_read = pd.read_csv(ftrain_labels).values.ravel()
                print('Train_labels - %s: Found File' % ftrain_labels)
            else:
                print('Train_labels - %s: File not exists' % ftrain_labels)
                sys.exit()
        elif task == 1:
            if os.path.exists(ftrain_labels):
                train_labels_read = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(nameseq_train)["nameseq"].to_list()]
                print('Train_labels - %s: Found File' % ftrain_labels)
            else:
                print('Train_labels - %s: File not exists' % ftrain_labels)
                sys.exit()

    test_read = ''
    if ftest:
        if os.path.exists(ftest):
            test_read = pd.read_csv(ftest)
            print('Test - %s: Found File' % ftest)
        else:
            print('Test - %s: File not exists' % ftest)
            sys.exit()

    test_nameseq_read = ''
    if nameseq_test:
        if os.path.exists(nameseq_test):
            test_nameseq_read = pd.read_csv(nameseq_test).values.ravel()
            print('Test_nameseq - %s: Found File' % nameseq_test)
        else:
            print('Test_nameseq - %s: File not exists' % nameseq_test)
            sys.exit()

    test_labels_read = ''
    if ftest_labels:
        if task == 0:
            if os.path.exists(ftest_labels):
                test_labels_read = pd.read_csv(ftest_labels).values.ravel()
                print('Test_labels - %s: Found File' % ftest_labels)
            else:
                print('Test_labels - %s: File not exists' % ftest_labels)
                sys.exit()
        elif task == 1:
            if os.path.exists(ftest_labels):
                test_labels_read = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(nameseq_test)["nameseq"].to_list()]
                print('Test_labels - %s: Found File' % ftest_labels)
            else:
                print('Test_labels - %s: File not exists' % ftest_labels)
                sys.exit()

    predictive_pipeline(
        model, task, tuning, train_read, train_labels_read, train_nameseq_read, 
        test_read, test_labels_read, test_nameseq_read, classifier, foutput
    )

    cost = (time.time() - start_time) / 60
    print('Computation time - Pipeline: %s minutes' % cost)
##########################################################################
##########################################################################
