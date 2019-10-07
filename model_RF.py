# Make necessary imports
from sklearn.ensemble import RandomForestClassifier

# Need to remove this but for now I'm not sure how to use the output of another python file as the input to this one (namely to get X_train, X_test, y_train and y_test directly here)
from split import split
from preprocess import preprocess
file_path = 'data/CS1_Interpretability.csv'
df = preprocess(file_path)

# Ideally start from here

def Get_Random_Forest(df):
    # Need to determine if still need to import the data this way
    # Split the dataset
    X_train, X_test, y_train, y_test = split(df)
    
    # Create the Model

    # Final Parameters obtained by performing RAndomSearchCV to locate roughly the best model, and performing GridSearch on a more restricted scope around the results from the random search.

    RFC = RandomForestClassifier(
            bootstrap=False, class_weight=None, criterion='gini',
            max_depth=80, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=2100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    
    # Return the Trained Model
    return RFC


### Interpretation

# Sort of Waterfall plot, extremely demanding in computing time though

import shap
explainer = shap.TreeExplainer(RFC)
shap_values = explainer.shap_values(X_test)
shap.sumary_plot(shap_values)
shap.force_plot(explainer.expected_value, shap_values[j], X_test.iloc[[j]])
