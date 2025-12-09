from sklearn.pipeline import Pipeline # Chains multiple transformations and a model together 
from sklearn.compose import ColumnTransformer # Allows for different preprocessing to different features
from sklearn.impute import SimpleImputer # Fills missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler # OHE converts categorical variables to numerical format for ML, SS does Z-score normalisation

num_cols = [...] # list of numerical features e.g. tenure, MonthlyCharges
cat_cols = [...] # list of categorical features e.g. gender, contract

