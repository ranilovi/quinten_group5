# Necessary Imports

import pandas as pd 

#%%
# Preprocessing Functions

def preprocess(df, drop_corr=False, onehot=False):
    '''
    Complete the entire preprocessing pipeline of the raw dataset
    Input:
        file_path : string
        
    Output:
        df : DataFrame
    '''
    if drop_corr==True: 
        df = drop_corr_columns (df)
    
    if onehot==True: 
        df = onehot_encode (df)
    return df


def load_csv(file_path):
    """Import dataset from csv, rename target varable to 'Target' and transform it to binary """

    df = pd.read_csv(file_path, sep = ';')
    df = df.rename(columns = {'TRUE_APPETENCE_Epargne_liquide' : 'Target'})
    df.Target = df.Target.astype(int)
    
    return df
#%%
def drop_corr_columns (df):
    """Remove highly correlated variables from the dataset"""
    df = df.drop(axis=1, labels=["Var_135", "Var_138", "Var_149", "Var_151", "Var_156","Var_157", "Var_159", 
                          "Var_162","Var_164", "Var_170", "Var_193", "Var_198", "Var_206", "Var_209",
                         "Var_21", "Var_213", "Var_218", "Var_219", "Var_223", "Var_29", "Var_30", 
                         "Var_39", "Var_42", "Var_43", "Var_48", "Var_60", "Var_62", "Var_64", "Var_65", 
                         "Var_78", "Var_8", "Var_82", "Var_89", "Var_9", "Var_96"])
    return df

def onehot_encode (df):
    '''Onehot encode categorical variables. Return df with categorical vars on the left'''
    
    
    categorical = ["Var_108", "Var_116", "Var_119", "Var_129", "Var_130", "Var_131",
                   "Var_141", "Var_142", "Var_154", "Var_174", "Var_184", "Var_194",
                   "Var_197", "Var_199", "Var_200",
                   "Var_22", "Var_28", "Var_40", "Var_49", "Var_57","Var_66", 
                   "Var_70", "Var_80", "Var_90", "Var_95", "Var_98", "Var_99"]
    #one hot encode categorical variables
    encoded = pd.get_dummies(df[categorical].astype(str), drop_first = True)
    
    #attach newly encoded features to the left of the numerical variables
    df = pd.concat([encoded, df.drop(categorical, axis = 1)], axis = 1)
    
    return df
    
   
    
    
    
    
    
    
    
    
