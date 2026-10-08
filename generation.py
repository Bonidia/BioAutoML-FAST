from bioautoml.model_artifacts import validate_model_support
from bioautoml.sequence_names import read_names, annotate_predictions
from bioautoml.execution import run_path, timed, start_cli
from bioautoml.execution import phase, run_root
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
import argparse
import json
import sys
import os.path
import time
import lightgbm as lgb
import joblib
from bioautoml.model_artifacts import get_training_count, load_model, save_model
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score
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
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import make_scorer, matthews_corrcoef, cohen_kappa_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from numpy.random import default_rng
from functools import partial
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score
from bioautoml.feature_execution import get_available_cpus
from bioautoml.homology import get_homology_folds, add_homology_arguments, resolve_homology_arguments, validate_report_settings

random_seed = 63
optimization_seed = 63

PEARSON_REASONS = (
    'ok', 'fewer than two observations', 'constant targets',
    'constant predictions', 'near-constant targets', 'near-constant predictions'
)


def pearson_score(y_true, y_pred):
    """Return signed Pearson r and a reason when correlation is undefined.

    Scale before centering to avoid overflow. Near-constant inputs are treated
    conservatively as undefined at a relative float64 tolerance of eps**0.75.
    Invalid input raises rather than silently changing evaluation membership.
    """
    arrays = [np.asarray(values, dtype=np.float64) for values in (y_true, y_pred)]
    if any(values.ndim != 1 for values in arrays):
        raise ValueError('Pearson requires one-dimensional targets and predictions.')
    if len(arrays[0]) != len(arrays[1]):
        raise ValueError('Pearson targets and predictions have different lengths.')
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError('Pearson requires finite targets and predictions.')
    if len(arrays[0]) < 2:
        return np.nan, PEARSON_REASONS[1]
    normalized = []
    for values, name in zip(arrays, ('targets', 'predictions')):
        if np.all(values == values[0]):
            return np.nan, f'constant {name}'
        values = values / np.max(np.abs(values))
        mean = np.mean(values)
        centered = values - mean
        norm = np.linalg.norm(centered)
        if norm <= np.finfo(np.float64).eps ** 0.75 * abs(mean):
            return np.nan, f'near-constant {name}'
        normalized.append(centered / norm)
    return float(np.clip(np.dot(*normalized), -1.0, 1.0)), PEARSON_REASONS[0]


def regression_scores(model, X, y):
    """Score all regression metrics from one prediction call per CV fold."""
    predictions = model.predict(X)
    pearson, reason = pearson_score(y, predictions)
    return {
        'MAE': -mean_absolute_error(y, predictions),
        'MSE': -mean_squared_error(y, predictions),
        'RMSE': -root_mean_squared_error(y, predictions),
        'R2': r2_score(y, predictions),
        'Pearson': pearson,
        'Pearson_status': PEARSON_REASONS.index(reason),
    }


def validate_feature_alignment(train, train_labels, train_nameseq, test, test_labels, test_nameseq, columns):
    """Validate parallel row metadata and enforce the saved feature order."""
    if len(train) != len(train_labels):
        raise ValueError('Training features and labels have different row counts.')
    if len(train_nameseq) and len(train_nameseq) != len(train):
        raise ValueError('Training features and sequence IDs have different row counts.')
    if len(train_nameseq) and pd.Index(train_nameseq).has_duplicates:
        raise ValueError('Training sequence IDs must be unique.')
    if train.columns.has_duplicates or pd.Index(columns).has_duplicates or list(train.columns) != list(columns):
        raise ValueError('Training feature columns do not match the model schema.')
    return validate_test_alignment(test, test_labels, test_nameseq, columns)


def validate_test_alignment(test, test_labels, test_nameseq, columns):
    """Inference needs the saved schema, not the original training matrix."""
    if pd.Index(columns).has_duplicates:
        raise ValueError('Model feature columns must be unique.')
    if not isinstance(test, pd.DataFrame):
        return test
    if len(test_nameseq) != len(test) or pd.Index(test_nameseq).has_duplicates:
        raise ValueError('Test sequence IDs must be unique and match the feature rows.')
    if len(test_labels) and len(test_labels) != len(test):
        raise ValueError('Test features and labels have different row counts.')
    if test.columns.has_duplicates or set(test.columns) != set(columns):
        raise ValueError('Test feature columns do not match the training schema.')
    return test.loc[:, columns].copy()


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
        "MAE", "MSE", "RMSE", "R2", "Pearson"
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
        if metric == 'Pearson':
            results['Pearson_valid_folds'] = int(np.isfinite(values).sum())
            results['Pearson_total_folds'] = len(values)

    # Build DataFrame (single-row)
    df = pd.DataFrame([results])

    # Write to CSV
    df.to_csv(
        output_measures,
        index=False,
    )

def get_cpu_parameters(n_cpu, requested_search_jobs=None):
    """Return total CPUs, Optuna workers, and threads available to each trial."""

    total_cpus = get_available_cpus(n_cpu)
    if requested_search_jobs is None:
        search_jobs = min(16, total_cpus)
    else:
        search_jobs = min(max(1, requested_search_jobs), total_cpus)
    model_jobs = max(1, total_cpus // search_jobs)
    return total_cpus, search_jobs, model_jobs

@timed('optimization_cv')
def evaluate_model_cross(X, y, model, task, output_cross, matrix_output, folds=None):
    """Run 10-fold cross-validation and write metrics and confusion matrix to CSV.

    task=0: classification — binary uses Sn/Sp/AUC/gmean/MCC; multiclass uses macro metrics.
    task=1: regression — writes MAE/MSE/RMSE/R2/Pearson; skips confusion matrix.
    Confusion matrix rows/columns are decoded through the global lb_encoder.
    """

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

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
        if folds is None:
            folds = list(kfold.split(X, y))
        scores = cross_validate(model, X, y, cv=folds, scoring=scoring, return_estimator=True)

        save_measures(output_cross, scores)

        y_pred = np.empty_like(np.asarray(y))
        for fitted_model, (_, validation_idx) in zip(scores["estimator"], folds):
            y_pred[validation_idx] = fitted_model.predict(X.iloc[validation_idx])

        conf_mat = pd.crosstab(
            lb_encoder.inverse_transform(y),
            lb_encoder.inverse_transform(y_pred),
            rownames=['REAL'], colnames=['PREDICTED'], margins=True
        )

        conf_mat.to_csv(matrix_output)
    else:
        kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
        scores = cross_validate(model, X, y, cv=kfold if folds is None else folds,
                                scoring=regression_scores, error_score='raise')
        reasons = [PEARSON_REASONS[int(code)] for code in scores.pop('test_Pearson_status')]
        pearson = scores['test_Pearson']
        pd.DataFrame({
            'fold': np.arange(1, len(pearson) + 1), 'Pearson': pearson,
            'Pearson_r2': pearson ** 2, 'reason': reasons,
        }).to_csv(os.path.splitext(output_cross)[0] + '_pearson_folds.csv', index=False)

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

    preds_df = annotate_predictions(preds_df, run_root(pred_output))
    preds_df.to_csv(pred_output, index=False)

def get_best_model_optuna(X, y, task, n_trials, n_cpu=-1, search_jobs=None, folds=None):
    """
    Runs Optuna optimization and returns the best configured pipeline.
    task: 0 = Classification, 1 = Regression
    classifier_type: 0 = RF, 1 = XGB, 2 = LGBM 
    """

    if isinstance(y, list):
        y = np.array(y)

    total_cpus, search_jobs, model_jobs = get_cpu_parameters(n_cpu, search_jobs)
    
    def objective(trial):
        # Define base pipeline components
        imputer = SimpleImputer(strategy='mean')
        
        classifier_type = trial.suggest_categorical('Classifier', [0, 1, 2])

        # --- CLASSIFICATION (Task 0) ---
        if task == 0:
            if classifier_type == 0: # Random Forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth_rf', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': random_seed
                }
                model = RandomForestClassifier(n_jobs=model_jobs, **params)
                
            elif classifier_type == 1: # XGBoost
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth_xgb', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'random_state': random_seed,
                    'eval_metric': 'mlogloss' if len(np.unique(y)) > 2 else 'logloss'
                }
                model = xgb.XGBClassifier(n_jobs=model_jobs, **params)
                
            elif classifier_type == 2: # LightGBM
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'random_state': random_seed,
                    'verbosity': -1
                }
                model = lgb.LGBMClassifier(n_jobs=model_jobs, **params)

            clf_pipeline = Pipeline(steps=[("imputer", imputer), ("clf", model)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

            fold_scores = []
            for step, (train_idx, val_idx) in enumerate(cv.split(X, y) if folds is None else folds):
                # Split data
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit and Predict
                clf_pipeline.fit(X_train, y_train)
                preds = clf_pipeline.predict(X_val)
                
                # Calculate Metric (MCC)
                metric = matthews_corrcoef(y_val, preds)
                fold_scores.append(metric)
                
                # Report intermediate mean score to Optuna
                intermediate_value = np.mean(fold_scores)
                trial.report(intermediate_value, step)
                
                # Prune if this trial is performing poorly compared to others at this step
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(fold_scores)

        # --- REGRESSION (Task 1) ---
        elif task == 1:
            common_lgb_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'random_state': random_seed,
                'verbosity': -1,
                'n_jobs': model_jobs
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
            cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)

            fold_scores = []
            for step, (train_idx, val_idx) in enumerate(cv.split(X, y) if folds is None else folds):
                # Split data
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit and Predict
                reg_pipeline.fit(X_train, y_train)
                preds = reg_pipeline.predict(X_val)
                
                # Calculate Metric (RMSE)
                metric = root_mean_squared_error(y_val, preds)
                fold_scores.append(metric)
                
                # Report intermediate mean score to Optuna
                intermediate_value = np.mean(fold_scores)
                trial.report(intermediate_value, step)
                
                # Prune if this trial is performing poorly compared to others at this step
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(fold_scores)

    # Create Study
    if task == 0:
        direction = "maximize"
    else:
        direction = "minimize"

    if n_trials > 0:
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(
                multivariate=True, group=True, constant_liar=True,
                seed=optimization_seed
            )
        )
        study.optimize(objective, n_trials=n_trials, timeout=10_800, show_progress_bar=True, n_jobs=search_jobs)
        
        print(f"Best Trial Score: {study.best_value:.4f}")
        print("Best Params Raw:", study.best_params)

        # --- RECONSTRUCT BEST MODEL ---
        best_params = study.best_params.copy()
        
        # 1. Extract and remove the classifier selector key
        best_clf_type = best_params.pop('Classifier')

    else:
        best_clf_type = 2
    
    final_model = None
    
    if task == 0:
        if best_clf_type == 0: # Random Forest
            if 'max_depth_rf' in best_params:
                best_params['max_depth'] = best_params.pop('max_depth_rf')
            final_model = RandomForestClassifier(random_state=random_seed, n_jobs=total_cpus, **best_params if n_trials > 0 else {})
            
        elif best_clf_type == 1: # XGBoost
            if 'max_depth_xgb' in best_params:
                best_params['max_depth'] = best_params.pop('max_depth_xgb')
            
            metric = 'mlogloss' if len(np.unique(y)) > 2 else 'logloss'
            final_model = xgb.XGBClassifier(random_state=random_seed, eval_metric=metric, n_jobs=total_cpus, **best_params if n_trials > 0 else {})
            
        elif best_clf_type == 2: # LightGBM
            final_model = lgb.LGBMClassifier(random_state=random_seed, verbosity=-1, n_jobs=total_cpus, **best_params if n_trials > 0 else {})

    else: # Regression (Task 1)
        if best_clf_type == 0: # LGBM (RF Mode)
            final_model = lgb.LGBMRegressor(boosting_type='rf', random_state=random_seed, verbosity=-1, n_jobs=total_cpus, **best_params if n_trials > 0 else {})
            
        elif best_clf_type == 1: # LGBM (GBDT Mode)
            final_model = lgb.LGBMRegressor(boosting_type='gbdt', random_state=random_seed, verbosity=-1, n_jobs=total_cpus, **best_params if n_trials > 0 else {})
            
        elif best_clf_type == 2: # LGBM (Standard)
            final_model = lgb.LGBMRegressor(random_state=random_seed, verbosity=-1, n_jobs=total_cpus, **best_params if n_trials > 0 else {})

    final_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy='mean')), ("clf", final_model)])
    return final_pipeline, best_clf_type

def predictive_pipeline(model, task, tuning, train, train_labels, train_nameseq, test, test_labels, test_nameseq, output, n_cpu=-1, search_jobs=None, homology_report=None):
    """End-to-end training and prediction pipeline.

    When model=None: encodes labels, imputes missing values, runs Optuna hyperparameter search
    (tuning trials), trains the best pipeline, and saves the model dict to output/.
    When model is a pre-loaded dict: skips training and applies the saved encoders/imputer/clf directly.
    Writes cross-validation metrics, confusion matrix, feature importances, and test predictions to output/.
    task: 0 = Classification, 1 = Regression.
    """

    global clf, lb_encoder, ord_encoder

    if model:
        validate_model_support(model)

    for directory in ('model', 'results/metrics', 'results/predictions', 'results/descriptors'):
        os.makedirs(os.path.join(output, directory), exist_ok=True)

    if model:
        column_train = model["column_train"]
        test = validate_test_alignment(test, test_labels, test_nameseq, column_train)
    else:
        column_train = train.columns

        model_dict = {
            "train": train,
            "train_labels": train_labels,
            "column_train": column_train,
            "feature_schema_version": 1,
        }
    
    if not model:
        test = validate_feature_alignment(train, train_labels, train_nameseq, test, test_labels, test_nameseq, column_train)
    search_folds, reporting_folds = None, None
    if not model:
        if homology_report:
            search_folds = get_homology_folds(homology_report, train_nameseq, task, random_seed, 5)
            reporting_folds = get_homology_folds(homology_report, train_nameseq, task, random_seed, 10)
            model_dict['homology_report'] = homology_report
        overlap_path = run_path(output, 'homology', 'overlap_report.json')
        if os.path.isfile(overlap_path):
            with open(overlap_path) as handle:
                model_dict['overlap_report'] = json.load(handle)
    column_test = ''

    """Basic Info"""
    print(f'Number of samples (train): {get_training_count(model) if model else len(train)}')
    print(f'Number of features (train): {len(column_train)}')

    if os.path.exists(ftest):
        column_test = test.columns
        print(f'Number of samples (test): {len(test)}')
        print(f'Number of features (test): {len(column_test)}')

    """Preprocessing:  Label Encoding"""

    if model:
        if "label_encoder" in model:
            lb_encoder = model["label_encoder"]

        if "ordinal_encoder" in model:
            ord_encoder = model["ordinal_encoder"]
            # Prediction uses only test rows; do not transform or mutate the
            # training frame retained for the web app's model inspection.
            if os.path.exists(ftest) is True:
                string_cols = test.select_dtypes(include=["object"]).columns
                if not string_cols.empty:
                    test[string_cols] = ord_encoder.transform(test[string_cols])
    else:
        lb_encoder, ord_encoder = LabelEncoder(), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

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
        with phase('stage_2', 'skipped' if tuning == 0 else 'completed'):
            clf, selected_type_id = get_best_model_optuna(train, train_labels, task, tuning, n_cpu, search_jobs, folds=search_folds)

        print('--- Optimization Complete ---')

        if task == 0:
            names = {0: "Random Forest", 1: "XGBoost", 2: "LightGBM"}
            print(f"Optuna Selected Classifier: {names[selected_type_id]}")
        else:
            names = {0: "LightGBM (RF Mode)", 1: "LightGBM (Random Hist)", 2: "LightGBM (Standard)"}
            print(f"Optuna Selected Regressor: {names[selected_type_id]}")

        print('--- Optimization Complete ---')
        
    """Training - StratifiedKFold (cross-validation = 10)..."""

    print('Training: post-selection homology-aware CV (10 folds, not nested)...' if homology_report and not model
          else 'Training: StratifiedKFold (cross-validation = 10)...')
    
    train_output = run_path(output, 'training_kfold(10)_metrics.csv')
    matrix_output = run_path(output, 'training_confusion_matrix.csv')
    importance_output = run_path(output, 'feature_importance.tsv')
    descriptors_output = run_path(output, 'best_descriptors/selected_descriptors.csv')
    model_output = run_path(output, 'trained_model.sav')

    if model:
        clf = model["clf"]
    else:
        evaluate_model_cross(train, train_labels, clf, task, train_output, matrix_output, reporting_folds)
        with phase('final_fit'):
            clf.fit(train, train_labels)
        model_dict["clf"] = clf

        model_dict["cross_validation"] = pd.read_csv(train_output)
        if task == 1:
            model_dict['pearson_folds'] = pd.read_csv(
                os.path.splitext(train_output)[0] + '_pearson_folds.csv')

        if task == 0:
            model_dict["confusion_matrix"] = pd.read_csv(matrix_output)

        if os.path.exists(descriptors_output):    
            model_dict["descriptors"] = pd.read_csv(descriptors_output)
        model_dict["nameseq_train"] = train_nameseq
        sequence_names = read_names(output)
        model_dict["sequence_names"] = {str(key): sequence_names[str(key)] for key in train_nameseq if str(key) in sequence_names}
        
        print('Saving results in ' + train_output + '...')
        print('Saving confusion matrix in ' + matrix_output + '...')
        print('Saving trained model in ' + model_output + '...')
        print('Training: Finished...')

        """Generating Feature Importance - Selected feature subset..."""

        print('Generating Feature Importance - Selected feature subset...')
        features_importance_ensembles(clf, column_train, importance_output)
        print('Saving results in ' + importance_output + '...')

        model_dict["feature_importance"] = pd.read_csv(importance_output, sep='\t')

        save_model(model_dict, model_output)

    """Testing model..."""

    if os.path.exists(ftest) is True:
        print('Generating Performance Test...')

        if task == 0:
            with phase('prediction'):
                preds = lb_encoder.inverse_transform(clf.predict(test))
                probs = clf.predict_proba(test)
            pred_output = run_path(output, "test_predictions.csv")
            print('Saving prediction in ' + pred_output + '...')
            save_prediction(task, probs, test_nameseq, pred_output)
        else:
            with phase('prediction'):
                preds = clf.predict(test)
            pred_output = run_path(output, 'test_predictions.csv')
            save_prediction(task, preds, test_nameseq, pred_output)

        if os.path.exists(ftest_labels) is True and len(np.unique(test_labels)) > 1:
            print('Generating Metrics - Test set...')
            
            if task == 0:
                report = classification_report(test_labels, preds, output_dict=True)

                metrics_output = run_path(output, "metrics_test.csv")
                print('Saving Metrics - Test set: ' + metrics_output + '...')
                
                metr_report = pd.DataFrame(report).transpose()
                metr_report.to_csv(metrics_output)
                
                if len(lb_encoder.classes_) <= 2:
                    metrics_other_output = run_path(output, "metrics_other.csv")
                    accu = accuracy_score(test_labels, preds)
                    auc = roc_auc_score(test_labels, probs[:, 1])
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
                matrix_output_test = run_path(output, "test_confusion_matrix.csv")
                matrix_test.to_csv(matrix_output_test)
                print('Saving confusion matrix in ' + matrix_output_test + '...')
                print('Task completed - results generated in ' + output + '!')
            elif task == 1:
                MAE = mean_absolute_error(test_labels, preds)
                MSE = mean_squared_error(test_labels, preds)
                RMSE = root_mean_squared_error(test_labels, preds)
                R2 = r2_score(test_labels, preds)
                pearson, reason = pearson_score(test_labels, preds)
                metrics = pd.DataFrame({
                    "Metric": ["MAE", "MSE", "RMSE", "R2", "Pearson"],
                    "Value": [MAE, MSE, RMSE, R2, pearson]
                })
                metrics_output = run_path(output, 'metrics_test.csv')
                metrics.to_csv(metrics_output, index=False)
                pd.DataFrame({'Pearson': [pearson], 'Pearson_r2': [pearson ** 2],
                              'reason': [reason], 'n_observations': [len(test_labels)]}).to_csv(
                    run_path(output, 'metrics_test_pearson.csv'), index=False)
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
    parser.add_argument('--trust_unsigned_model', action='store_true', help='LOCAL ONLY: explicitly trust an unsigned current-format model; loading can execute code')
    parser.add_argument('-task', '--task', default=0, help='Machine learning task - 0: Classification, 1: Regression - Default: Classification')
    parser.add_argument('-tuning', '--tuning', default=150, help='number of trials for hyperparameter tuning - default = 150')
    parser.add_argument('-train', '--train', help='csv format file, e.g., train.csv')
    parser.add_argument('-train_label', '--train_label', default='', help='csv format file, e.g., labels.csv')
    parser.add_argument('-train_nameseq', '--train_nameseq', default='', help='csv with sequence names')
    parser.add_argument('-test', '--test', default='', help='csv format file, e.g., test.csv')
    parser.add_argument('-test_label', '--test_label', default='', help='csv format file, e.g., labels.csv')
    parser.add_argument('-test_nameseq', '--test_nameseq', default='', help='csv with sequence names')
    parser.add_argument('-n_cpu', '--n_cpu', default=-1, help='number of cpus - default = all')
    parser.add_argument('-search_jobs', '--search_jobs', default=1, help='parallel Optuna workers; default 1 for repeatable trial ordering')
    parser.add_argument('--homology_aware', action='store_true', help='Use the automatic assignments prepared by engineering.py in output/homology; raw sequences are required upstream')
    add_homology_arguments(parser)
    parser.add_argument('-seed', '--seed', default=63, help='random seed for cross-validation and learners - default = 63')
    parser.add_argument('-search_seed', '--search_seed', default=None, help='Optuna sampler seed; defaults to --seed')
    parser.add_argument('-output', '--output', required=True, help='results directory, e.g., result/')
    args = parser.parse_args()
    resolve_homology_arguments(parser, args)
    if args.homology_aware and args.path_model:
        parser.error('--homology_aware is a training option, not an inference option.')
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
    search_jobs = int(args.search_jobs) if args.search_jobs is not None else None
    random_seed = int(args.seed)
    optimization_seed = int(args.search_seed) if args.search_seed is not None else random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    foutput = str(args.output)
    execution = start_cli(args, 'prediction' if path_model else 'training')
    start_time = time.time()

    model = ''
    train_read, train_labels_read, train_nameseq_read = '', '', ''
    if path_model:
        if args.trust_unsigned_model and os.environ.get('BIOAUTOML_WEB_EXECUTION') == '1':
            parser.error('Unsigned-model loading is forbidden in web jobs.')
        model = load_model(path_model, trust_unsigned=args.trust_unsigned_model)
        if execution.owner:
            execution.record['model_id'] = getattr(model, 'model_id', None)
    else:
        if os.path.exists(ftrain):
            train_read = pd.read_csv(ftrain)
            print('Train - %s: Found File' % ftrain)
        else:
            print('Train - %s: File not exists' % ftrain)
            raise FileNotFoundError('Required input file is missing; see the preceding message.')

        if os.path.exists(nameseq_train):
            train_nameseq_read = pd.read_csv(nameseq_train).values.ravel()
            print('Train_nameseq - %s: Found File' % nameseq_train)
        else:
            print('Train_nameseq - %s: File not exists' % nameseq_train)
            raise FileNotFoundError('Required input file is missing; see the preceding message.')

        if task == 0:
            if os.path.exists(ftrain_labels):
                train_labels_read = pd.read_csv(ftrain_labels).values.ravel()
                print('Train_labels - %s: Found File' % ftrain_labels)
            else:
                print('Train_labels - %s: File not exists' % ftrain_labels)
                raise FileNotFoundError('Required input file is missing; see the preceding message.')
        elif task == 1:
            if os.path.exists(ftrain_labels):
                train_labels_read = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(nameseq_train)["nameseq"].to_list()]
                print('Train_labels - %s: Found File' % ftrain_labels)
            else:
                print('Train_labels - %s: File not exists' % ftrain_labels)
                raise FileNotFoundError('Required input file is missing; see the preceding message.')

    test_read = ''
    if ftest:
        if os.path.exists(ftest):
            test_read = pd.read_csv(ftest)
            print('Test - %s: Found File' % ftest)
        else:
            print('Test - %s: File not exists' % ftest)
            raise FileNotFoundError('Required input file is missing; see the preceding message.')

    test_nameseq_read = ''
    if nameseq_test:
        if os.path.exists(nameseq_test):
            test_nameseq_read = pd.read_csv(nameseq_test).values.ravel()
            print('Test_nameseq - %s: Found File' % nameseq_test)
        else:
            print('Test_nameseq - %s: File not exists' % nameseq_test)
            raise FileNotFoundError('Required input file is missing; see the preceding message.')

    test_labels_read = ''
    if ftest_labels:
        if task == 0:
            if os.path.exists(ftest_labels):
                test_labels_read = pd.read_csv(ftest_labels).values.ravel()
                print('Test_labels - %s: Found File' % ftest_labels)
            else:
                print('Test_labels - %s: File not exists' % ftest_labels)
                raise FileNotFoundError('Required input file is missing; see the preceding message.')
        elif task == 1:
            if os.path.exists(ftest_labels):
                test_labels_read = pd.read_csv(ftest_labels).values.ravel()
                if "Predicted" not in test_labels_read:
                    test_labels_read = [float(nameseq.split("|")[-1]) for nameseq in pd.read_csv(nameseq_test)["nameseq"].to_list()]
                print('Test_labels - %s: Found File' % ftest_labels)
            else:
                print('Test_labels - %s: File not exists' % ftest_labels)
                raise FileNotFoundError('Required input file is missing; see the preceding message.')

    homology_report = None
    if args.homology_aware:
        if path_model:
            parser.error('--homology_aware is a training option, not an inference option.')
        report_path = run_path(foutput, 'homology', 'homology_report.json')
        if not os.path.isfile(report_path):
            parser.error('Run engineering.py --homology_aware with FASTA inputs first; feature tables alone cannot establish homology.')
        with open(report_path) as handle:
            homology_report = json.load(handle)
        if not homology_report.get('enabled'):
            parser.error('No enabled homology-aware assignments were found.')
        try:
            validate_report_settings(homology_report, args.homology_identity, args.homology_coverage)
        except (KeyError, ValueError) as error:
            parser.error(str(error))
    predictive_pipeline(
        model, task, tuning, train_read, train_labels_read, train_nameseq_read, 
        test_read, test_labels_read, test_nameseq_read, foutput, n_cpu,
        search_jobs, homology_report=homology_report
    )

    cost = (time.time() - start_time) / 60
    print('Computation time - Pipeline: %s minutes' % cost)
    execution.finish()
##########################################################################
##########################################################################
