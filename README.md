# model_interpretability_quinten

General files:
*  *Notebook Interpretation.ipynb* => reference notebook given on 02/10/2019
*  *.gitignore* => gathering all the files that are not relevant for git follow-up  

.py files:
*  *preprocess.py* => file gathering all the functions for data loading, cleaning and one-hot-encoding
*  *split.py* => file to split the data, with a set seed 
*  *model_LR.py* => file for training and fitting a Linear Regression model
*  *model_XGboost.py* => file for training and fitting an XGBoost model
*  *model_LGBM.py* => file for training and fitting an LGBM model
*  *eval.py* => file gathering all the functions for models' predictions and performance assessments
*  *interpret.py* => file gathering all the functions for models interpretation **UPCOMING**   

Files to be deleted: 
*  *main.py* => not useful at this stage
*  *LightGBM.ipynb* => to be deleted ultimately (now uncapsulated in model_LGBM.py)
*  *Notebook Interpretation.ipynb* => study file to explore interpretability methods for LightGBM
