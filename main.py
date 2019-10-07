#%%
# Execution Flow
from preprocess import load_csv
from split import split

# Set the path to the raw data
FILE_PATH = ('data/CS1_Interpretability.csv')


# Preprocess the data and return dataframe or numpy array depending on model needs
df = load_csv(FILE_PATH)

