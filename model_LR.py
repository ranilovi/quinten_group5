# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics 

import scipy
from scipy import stats
from math import sqrt
from split import split
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess
import shap


#%%
def mean_confidence_interval(data, confidence=0.95):
    """Compute mean and conf interval"""
    a = 1.0*np.array(data)
    n = len(a)
    mu,sd = np.mean(a),np.std(a)
    z = stats.t.ppf(confidence, n)
    h=z*sd/sqrt(n)
    return mu, h

def get_avg_cm(cms):
    """get average confusion matrix"""
    tpr, fnr, fpr, tnr = [],[],[],[]
    for c in cms.values():
        c = c/c.sum()
        tnr.append(c[0][0])
        fpr.append(c[1][0])
        fnr.append(c[0][1])
        tpr.append(c[1][1])
    cm = pd.DataFrame({'Act_0': [np.mean(tnr), np.mean(fpr)], 'Act_1' : [np.mean(fnr), np.mean(tpr)]}, index = ['Pred_0','Pred_1'])
    return cm
#%%
def logistic_regression(df, num_run = 5, **params):
    '''
    train and test a logistic regression model
    
    Input
    df: DataFrame. Preprocessed data
    num_run: int. How many times you want to run for random evaluation?
    params: string->real. Hyper-parameter of classifier, inverse of regularization strength. Ex. c=1.0
    
    Output
    train_scores: list. Results of trails
    test_scores: list. Results of trails
    train_mean: scalar. Average accuracy
    test_mean: scalar. Average accuracy
    train_ci: scalar. Confidence Interval
    test_ci: scalar. Confidence Interval
    cm: confusion matrix
    '''
    
    classifier = LogisticRegression(solver='newton-cg', max_iter=1000, C = c, penalty = 'l2')
    
    train_scores=[]
    test_scores=[]
    
    cms={}
    labels = df['Target'].unique()
    
    for i in range (num_run):
        # separate datasets into training and test datasets once0
        X_train, X_test, y_train, y_test = split (df, percent_train = .75, frames = None)
        
        # train the features and target datasets and fit to a model
        clfModel = classifier.fit(X_train, y_train)
        
        # predict target with feature test set using trained model
        y_pred_train = list(clfModel.predict(X_train))
        y_pred_test = list(clfModel.predict(X_test))

        train_scores.append(metrics.f1_score(y_train, y_pred_train))
        test_scores.append(metrics.f1_score(y_test, y_pred_test))
        
        cms[i]=(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_test), columns=labels, index=labels))
       
    train_mean = np.mean(train_scores)
    test_mean = np.mean(test_scores)

    train_ci = mean_confidence_interval (train_scores,0.95) [1]
    test_ci =  mean_confidence_interval (test_scores,0.95) [1]
    
    
    return train_scores,test_scores,train_mean,test_mean,train_ci,test_ci, cms
#%%
def hyperParameterLR(df, num_run = 10,  params = np.logspace(-4, 4, num=9)):
    """
    INPUT
        df: Dataframe. Pre-processed data
    
    OUTPUT
        test_mean_hp: list.  mean accuracy list of test
        test_ci_hp: list. confidence interval list of test
        train_mean_hp: list. mean accuracy list of train
        train_ci_hp: list. confidence interval list of train
    """
    test_mean_hp=[]
    test_ci_hp=[]
    train_mean_hp=[]
    train_ci_hp=[]    
    

    for p in params:
        train_scores,test_scores,train_mean,test_mean,train_ci,test_ci,cms = logistic_regression(df, num_run, c=p)
         
        test_mean_hp.append(test_mean)
        test_ci_hp.append(test_ci)
  
        train_mean_hp.append(train_mean)
        train_ci_hp.append(train_ci)  
    
    return train_mean_hp, train_ci_hp, test_mean_hp, test_ci_hp

def hyperParameterPlot(params, train_mean_hp, train_ci_hp, test_mean_hp, test_ci_hp, num_run):
    # First illustrate basic pyplot interface, using defaults where possible.
    plt.figure()
    test_curve=plt.errorbar(params, test_mean_hp, color=sns.xkcd_rgb["pale red"], yerr=test_ci_hp)
    train_curve=plt.errorbar(params, train_mean_hp,color=sns.xkcd_rgb["denim blue"], yerr=train_ci_hp)
    plt.legend([test_curve, train_curve], ['Test', 'Train'])
    plt.xlabel('Parameter')
    plt.xscale("log")
    plt.ylabel('F1')
    plt.title("F1 score vs Parameters: {} runs".format(num_run))
    plt.show()
    
#%%
def eval_LR(c = 4, num_run = 10):
    train_scores,test_scores,train_mean,test_mean,train_ci,test_ci, cms = logistic_regression(df, num_run, c = c)
    
    print("Train\
        \nAverage F1: {0} \
        \nConfidence Interval: {1}\n".format(train_mean, train_ci)
         )
    print("Test\
        \nAverage F1: {0} \
        \nConfidence Interval: {1}".format(test_mean, test_ci)
     )
    cm = get_avg_cm(cms)
    sns.heatmap(cm, annot = True)

def grid_search_LR(df, grid = np.linspace(.5,10,20), num_run = 10):#np.logspace(-4, 4, num=9)    
    train_mean_hp, train_ci_hp, test_mean_hp, test_ci_hp = hyperParameterLR(df,num_run= num_run, params = grid)
    hyperParameterPlot(grid,train_mean_hp, train_ci_hp, test_mean_hp, test_ci_hp, num_run)


#%%
#define the model   
c = 4
model = LogisticRegression(solver='newton-cg', max_iter=1000, C = 4, penalty = 'l2' )

#split the data
df_preprocessed = preprocess(df, True, True)
X_train, X_test, y_train, y_test = split (df_preprocessed, percent_train = .75, frames = False, rs = None)

#train the model
model = model.fit(X_train, y_train)

#get greatest coefficients
variables = pd.DataFrame({'variable' :df_preprocessed.columns[:-1], 'coefficient' :list(model.coef_)[0]})
variables['exp']=variables.coefficient.apply(np.exp)
print("Top 10 variables : 'appetant'")
print(variables.sort_values('coefficient', ascending = False)[:15])
print("Top 10 variables : 'not appetant'")
print(variables.sort_values('coefficient', ascending = True)[:15])

#predict class and probability
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

#logit transformation of probability prediciton
y_pred_odds = scipy.special.logit(y_pred_proba)
matrix = pd.DataFrame((X_test*model.coef_).transpose(), index = df_preprocessed.columns[:-1])
predicitons = pd.DataFrame({'probability':y_pred_proba[:,1], 'class':y_pred, 'odds': y_pred_odds[:,1]})
#%%
def most_influential(pred_index):
    print ("Prediction: ")
    print (predicitons.iloc[pred_index])
    
    print ('Most inflential variables: ')
    if predicitons.iloc[pred_index]['class'] == 1:
        print(matrix[pred_index].nlargest(10))
    else: print(matrix[pred_index].nsmallest(10))
        
         
most_influential(4)  
most_influential(8)    

#%%
from IPython.display import display, HTML
explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_test)
X_test_array = X_test # we need to pass a dense version for the plotting functions

shap.summary_plot(shap_values, X_test_array, feature_names=df_preprocessed.columns[:-1])

obj = shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:])


with open('C:/Users/dorar_000/Documents/GitHub/model_interpretability_quinten/obj.htm','wb') as f:   # Use some reasonable temp name
    f.write(obj)





















