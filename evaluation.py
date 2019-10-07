# -*- coding: utf-8 -*-
#%%

# Custom functions
from split import split
from preprocess import preprocess
from model_XGB import *

# Necessary functions
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tabulate import tabulate
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
import lightgbm as lgb

#%%
def get_model(model_fam):
    """
    get model specified by model family
    Input
    model_fam : string. Model family - "LR", "RF", "XGB", or "GBM"
    
    Output
    model : model. Preferably one .fit and .predict can be called on
    """
    model = None
    
    if model_fam == ("LR"):
        model = LogisticRegression(solver='newton-cg',
                                   max_iter=1000,
                                   C = 4,
                                   penalty = 'l2'
                                  )
    
    elif model_fam == ("RF"):
        model = RandomForestClassifier(
            bootstrap=False, class_weight=None, criterion='gini',
            max_depth=80, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=2100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


    elif model_fam == ("XGB"):
        model = xgb.XGBClassifier(base_score=0.5, 
                                  booster='gbtree', 
                                  colsample_bylevel=1,
                                  colsample_bynode=1,
                                  colsample_bytree=0.85,
                                  gamma=0.0,
                                  learning_rate=0.1,
                                  max_delta_step=0, 
                                  max_depth=9,
                                  min_child_weight=5,
                                  missing=None,
                                  n_estimators=140,
                                  n_jobs=1,
                                  nthread=4,
                                  objective='binary:logistic',
                                  random_state=0,
                                  reg_alpha=0,
                                  reg_lambda=1,
                                  scale_pos_weight=1,
                                  seed=27,
                                  silent=None,
                                  subsample=0.8,
                                  verbosity=1
                                  )

    elif model_fam == ("LGBM"):
        model = lgb.LGBMClassifier(boosting_type='gbdt',
                                  class_weight=None,
                                  colsample_bytree=0.64,
                                  device='cpu',
                                  importance_type='gain',
                                  learning_rate=0.1,
                                  max_bin=400,
                                  max_depth=15,
                                  metric='roc_auc',
                                  min_child_samples=22,
                                  min_child_weight=0.001,
                                  min_split_gain=0,
                                  n_estimators=100,
                                  n_jobs=-1,
                                  num_leaves=15,
                                  objective='binary',
                                  random_state=None,
                                  reg_alpha=0.1,
                                  reg_lambda=1,
                                  silent=True,
                                  subsample=0.63,
                                  subsample_for_bin=200000,
                                  subsample_freq=0
                                  )
    else:
        print ("Invalid input!! Choose 'LR', 'RF', 'XGB' or 'GBM'. ")
    return model


def compare_models(list_of_models, df, percent_train):
    """
    Outputs f1 score accuracy table for each of the different models.
    Args :
        list_of_models : models to choose 'LR', 'RF', 'XGB' or 'LGBM'
        df : original Data Frame
    Outputs:
        Table disp
        Confusion Matrix plot
        ROC curves plot
        model_predictions: model name:
                                predictions
        model_results: model name:
                                f1 score
                                accuracy
                                mean CV f1 score
                                median CV f1 score
    """
    # Split the data into trainingn and testing sets 
    
    model_result = {m :[] for m in list_of_models}
    model_predictions = {m :[] for m in list_of_models}
    model_proba_predictions = {m :[] for m in list_of_models}
    fitted_models = []
    
    # Output table with results
    for model_name in list_of_models:
        if model_name == 'LGBM':
            df_prep = preprocess(df,
                                 drop_corr=True,
                                 onehot=False
                                )
        else:
            df_prep = preprocess(df,
                                 drop_corr=True,
                                 onehot=True
                                )


        X_train, X_test, y_train, y_test = split(df, percent_train, frames = False)
        # Obtain models
        model = get_model(model_name)
        
        # Compute cross validation of models:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring = 'f1')
        model_result[model_name].append(scores.mean())
        model_result[model_name].append(np.median(scores))
        # Fit the models to the data set and time execution
        start_time = time.time()
        model = model.fit(X_train, y_train)
        total_time = time.time() - start_time
        
        # Store model in list
        fitted_models.append(model)
        
        # predict the labels of the test set
        y_pred = model.predict(X_test)
    
        # test the accuracy and f1 score for that particular model
        f1_score_model, accuracy_model = eval_metrics(y_test, y_pred)
        
        # Store the results of accuracy and f1 score in a dict with model name as key.
        model_result[model_name].extend([f1_score_model, accuracy_model])
        
        # Time exection
        model_result[model_name].append(total_time)
        
        # Store output predictions for each model in dict with model name as key.
        model_predictions[model_name] = y_pred
        
        #predict the probablibities of the labels of the test set and store them in a dict
        y_pred_proba = model.predict_proba(X_test)
        model_proba_predictions[model_name] = y_pred_proba
        
        # Compute average rank
        rank = avg_rank(y_pred_proba, y_test)
        model_result[model_name].append(rank)
    
    model_result['metric'] = ['mean_CV_f1_score', 'median_CV_f1_score', 'f1_score', 'accuracy', 'Training_time_execution','Avg Rank']
    print(tabulate(model_result,
                   headers="keys"
                  )
         )
        
    # Output Confusion matrix
    for model_name in list_of_models:
        plot_confusion_matrix(y_test, 
                              model_predictions[model_name],                        
                              np.array([0,1]),
                              model_name,
                              normalize=False,
                              title= 'confusion matrix ' + str(model_name) + ' model',
                              cmap=plt.cm.Blues
                             )
    
    # Plot list of models same plot
    plot_auc(model_predictions, model_proba_predictions, y_test)
    
    return model_predictions, model_result
#%%

def eval_metrics(y_test, y_pred):
    '''
    Outputs f1 score and accuracy_scores:
    Args:
        y_test: true target values
        y_pred: predicted target values
    Outputs:
        f1_score_model: f1 score
        accuracy_model: accuracy
    '''
    f1_score_model = f1_score(y_test, y_pred)
    accuracy_model = accuracy_score(y_test, y_pred)
    
    return f1_score_model, accuracy_model


def plot_confusion_matrix(y_true, 
                          y_pred,
                          classes,
                          model_name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues
                         ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim(1.5,-0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    confusion_matrix_directory = os.path.join(os.getcwd(), 'confusion_matrix')
    if not os.path.exists(confusion_matrix_directory):
        os.makedirs(confusion_matrix_directory)
    plt.savefig(os.path.join(confusion_matrix_directory, model_name))
    return ax   

def plot_auc(model_prediction, model_proba, y_test):
    plt.figure()
    for model_name in model_prediction:
        logit_roc_auc = roc_auc_score(y_test, model_prediction[model_name])
        fpr, tpr, thresholds = roc_curve(y_test, model_proba[model_name][:,1])
        plt.plot(fpr, tpr, label='{0} (area = {1})'.format(model_name, round(logit_roc_auc,2)))
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
    roc_auc_directory = os.path.join(os.getcwd(), 'roc_auc')
    if not os.path.exists(roc_auc_directory):
        os.makedirs(roc_auc_directory)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Customer appetancy')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(roc_auc_directory,'Log_ROC'))
    plt.show()

def avg_rank(y_pred_proba, y_test):
    
    table = pd.DataFrame({'proba':y_pred_proba[:,1], 'true': y_test}).sort_values(by = 'proba', ascending = False)
    table['rank'] = range(1, len(table)+1) 
    avg = table[table.true == 1]['rank'].mean()
    return round(avg)
    
