# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%% 

def split (df, percent_train = .75, frames = True, rs = 5):
    '''
    Split data into train and test and apply normalization
    
    Input
    df : DataFrame
    percent_train : float between 0 and 1
    
    Output
    X_train : ndarray
    X_test : ndarray
    y_train : ndarray
    y_test : ndarray
    '''
    #split
    X = df.loc[:, df.columns != 'Target']
    y = df.Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= percent_train, random_state = rs)
    
    #scale
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    if frames:
        # Pandas data frame
        X_columns = X.columns
        y_columns = ['Target']
        
        X_train = pd.DataFrame(X_train, columns = X_columns)
        X_test = pd.DataFrame(X_test, columns = X_columns)
        y_train = pd.DataFrame(y_train, columns = y_columns)
        y_test = pd.DataFrame(y_test, columns = y_columns)

    return X_train, X_test, y_train, y_test