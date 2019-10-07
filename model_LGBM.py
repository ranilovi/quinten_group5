# -*- coding: utf-8 -*-

# Necessary imports
import pandas as pd
import numpy as np
import time

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

# Prepares the default version of LGBM with some core parameters
# essential for this problem.
def prepare_LGBM(metric, boosting_type = 'gbdt'): 
    LGBM = lgb.LGBMClassifier(boosting_type = boosting_type,
        objective = 'binary',
        n_jobs = -1,
        silent = True,
        importance_type = 'gain',
        metric = metric,
        device = 'cpu')
    
    return LGBM

# Prepares the default version of GridSearch for classifiers with some core parameters
# essential for this problem.
def prepare_GS(gridParams, model, metric):
    gsearch = GridSearchCV(estimator=model,
                    param_grid=gridParams,
                    verbose=1,
                    cv=StratifiedKFold(3, True),
                    n_jobs=-1,
                    scoring=metric,
                    refit=True)
    return gsearch

# Inputs: X and y for training, classifier parameters to be searched,
# classifier, scoring metric {roc_auc in this case} 
def run_GS(X, y, gridParams, model, metric):
    gsearch = prepare_GS(gridParams, model, metric)
    lgbm_model = gsearch.fit(X, y)
    return lgbm_model.best_params_, lgbm_model.best_score_, lgbm_model.best_estimator_

# Definition of parameters to search through. Second iterration is generated around best previous result.
def get_test_params(i, model):
    if i:
        return {
            'boosting_type' : ['gbdt'],
            'objective' : ['binary'],
            'n_estimators' : np.append(np.array(model.get_params()['n_estimators']), np.linspace(model.get_params()['n_estimators']*0.85, model.get_params()['n_estimators']*1.15, 5, dtype='int')),
            'num_leaves' : np.append(np.array(model.get_params()['num_leaves']), np.linspace(model.get_params()['num_leaves']*0.85, model.get_params()['num_leaves']*1.15, 5, dtype='int')),
            'learning_rate' : np.append(np.array(model.get_params()['learning_rate']), np.linspace(model.get_params()['learning_rate']*0.5, model.get_params()['learning_rate']*1.5, 5)),
            'max_depth' : np.append(np.array(model.get_params()['max_depth']), np.linspace(model.get_params()['max_depth']*0.9, model.get_params()['max_depth']*1.5, 5, dtype='int')),
            'min_child_samples' : np.append(np.array(model.get_params()['min_child_samples']), np.linspace(model.get_params()['min_child_samples']*0.85, model.get_params()['min_child_samples']*1.15, 5, dtype='int')),
            'max_bin' : np.append(np.array(model.get_params()['max_bin']), np.linspace(model.get_params()['max_bin']*0.85, model.get_params()['max_bin']*1.15, 5, dtype='int')),
            'colsample_bytree' : np.append(np.array(model.get_params()['colsample_bytree']), np.linspace(model.get_params()['colsample_bytree']*0.94, model.get_params()['colsample_bytree']*1.06, 5)),
            'subsample' : np.append(np.array(model.get_params()['subsample']), np.linspace(model.get_params()['subsample']*0.94, model.get_params()['subsample']*1.06, 5)),
            'reg_alpha' : np.append(np.array(model.get_params()['reg_alpha']), np.linspace(model.get_params()['reg_alpha']*0.5, model.get_params()['reg_alpha']*1.5, 5)),
            'reg_lambda' : np.append(np.array(model.get_params()['reg_lambda']), np.linspace(model.get_params()['reg_lambda']*0.5, model.get_params()['reg_lambda']*1.5, 5)),
            'min_split_gain' : np.append(np.array(model.get_params()['min_split_gain']), np.linspace(model.get_params()['min_split_gain']/5, model.get_params()['min_split_gain']*5, 5))
            }
    else:
        return {
            'boosting_type' : ['gbdt'],
            'objective' : ['binary'],
            'n_estimators': [100, 115, 125, 140, 150, 170, 200],
            'num_leaves': [15, 19, 25, 28, 32],
            'learning_rate': [0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
            'max_depth' : [15, 18, 20, 22],
            'min_child_samples' : [18, 20, 22, 24],
            'max_bin':[350, 370, 400, 420, 450],
            'colsample_bytree' : [0.62, 0.63, 0.64, 0.65],
            'subsample' : [0.61, 0.62, 0.63, 0.64],
            'reg_alpha' : [0.15, 0.2, 0.25, 0.3],
            'reg_lambda' : [0, 0.1, 0.3, 0.5, 0.8, 0.9],
            'min_split_gain' : [0, 0.001, 0.0001, 0.00001]
            }

# Defines order in which the parameters are tuned. They are tuned in pairs (+ the last one alone).
def get_GS_order():
    return ['boosting_type', 'objective',
            'n_estimators', 'num_leaves',
            'learning_rate', 'max_depth',
            'min_child_samples', 'max_bin',
            'colsample_bytree', 'subsample',
            'reg_alpha', 'reg_lambda',
            'min_split_gain']

# Sequential grid search - parameters are tuned in pairs to decrease complexity.
# "repeat" defines how many parameters are tuned at the same time.
def train_LGBM(repeat, testParams, GS_order, X_train, y_train, model, metric):
    gridParams = {}

    for i in range(repeat):
        a = GS_order.pop(0)
        gridParams[a] = testParams.get(a)
        
    return run_GS(X_train, np.array(y_train).ravel(), gridParams, model, metric)

# Performs 3 full grid searches (full = all parameters tuned two-by-two). First with manually set grid search intervals.
# Second and third iterrations of grid search have generated parameters based on previous best result.
def get_trained_LGBM(X_train, y_train, metric = 'roc_auc'):
    model = prepare_LGBM(metric=metric)
    
    for i in [0,1,2]:
        testParams = get_test_params(i, model)
        GS_order = get_GS_order()
        while len(GS_order) >= 3:
            param, score, model = train_LGBM(2, testParams, GS_order, X_train, y_train, model, metric)
        param, score, model = train_LGBM(1, testParams, GS_order, X_train, y_train, model, metric)
    return param, score, model

# Main callable function.
# Output is trained model and total time of execution.
def LGBM_model(X_train, y_train, X_test, y_test):
    # Time model
    start_time = time.time()

    best_params, best_score, lgbm_model = get_trained_LGBM(X_train, y_train)
    print(lgbm_model)
    #ccc = CalibratedClassifierCV(lgbm_model, 'sigmoid', 5)
    #calibrated_lgbm_model = ccc.fit(X_train, np.array(y_train).ravel())
    
    total_time = time.time() - start_time
    #return calibrated_lgbm_model, total_time
    return lgbm_model, total_time