# -*- coding: utf-8 -*-

# Necessary imports
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV



def xgboost_model(X_train, X_test, y_train, y_test):
    '''
    Finds the hyperparameters for the XGBoost model using a grid search cross validation optimising the F1 score
    Inputs:
        X_train: Training set
        X_test: Testing set
        y_train: Training target
        y_test: Testing target
    Outputs:
        model: The optimised model with the fine tuned hyperparameters
    '''
    # Time model
    start_time = time.time()
    
    ##### TUNE THE SHAPE OF THE TREES ######
    # Tune max_depth: The maximum number of nodes that can exist between the root and the farthest leaf, smaller values prevent overfitting.
    # Tune min_child_weight: Defines the min sum of weights for the observations in a child, too high and it overfits, too low and it underfits.
    param_test1 = {'max_depth':range(3,10,2),
                   'min_child_weight':range(1,6,2)
                  }
    
    # Set the parameters for the grid search cross validation
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate = 0.1, 
                                                          n_estimators = 140,
                                                          max_depth = 5, 
                                                          min_child_weight = 1,
                                                          gamma = 0,
                                                          subsample=0.8, 
                                                          colsample_bytree=0.8,
                                                          objective= 'binary:logistic',
                                                          nthread=4,
                                                          scale_pos_weight=1,
                                                          seed=27), 
                            param_grid = param_test1,
                            scoring='f1',
                            n_jobs= -1,
                            iid=False,
                            cv=5
                           )
    
    # Perform the grid search cross validation on the data
    gsearch1.fit(X_train, y_train)
    
    # Select the hyperparameters that give the best F1 score
    model = gsearch1.best_estimator_
    
    ##### TUNING GAMMA #####
    # Tune gamma, defines the minimum loss for a split to be made at a node.
    param_test2 = {'gamma':[i/10.0 for i in range(0,5)]
                  }
    
    # Set the parameters for the grid search cross validation
    gsearch2 = GridSearchCV(estimator = model, 
                            param_grid = param_test2,
                            scoring='f1',
                            n_jobs=-1,
                            iid=False,
                            cv=5
                           )
    # Perform the grid search cross validation on the data
    gsearch2.fit(X_train, y_train)
    
    # Select the hyperparameters that give the best F1 score
    model = gsearch2.best_estimator_
    
    ##### TUNE THE TREE POPULATION #####
    # subsample: percentage of training data used to train tree, too low underfits too high overfits.
    # colsample_bytree: percentage of features to be used to train trees.
    param_test3 = {'subsample':[i/100.0 for i in range(75,90,5)],
                   'colsample_bytree':[i/100.0 for i in range(75,90,5)]
                  }

    gsearch3 = GridSearchCV(estimator = model, 
                            param_grid = param_test3,
                            scoring='f1_weighted',
                            n_jobs=-1,
                            iid=False,
                            cv=5
                           )
    
    # Perform the grid search cross validation on the data
    gsearch3.fit(X_train, y_train)
    
    # Select the hyperparameters that give the best F1 score
    model = gsearch3.best_estimator_
    total_time = time.time() - start_time
    return model, total_time
