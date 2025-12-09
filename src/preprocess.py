from sklearn.pipeline import Pipeline # Chains multiple transformations and a model together 
from sklearn.compose import ColumnTransformer # Allows for different preprocessing to different features
from sklearn.impute import SimpleImputer # Fills missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler # OHE converts categorical variables to numerical format for ML, SS does Z-score normalisation

num_cols = [...] # list of numerical features e.g. tenure, MonthlyCharges
cat_cols = [...] # list of categorical features e.g. gender, contract

numPipline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # If the numerical column has blank value, fill with mode
    ('scalar', StandardScaler()) # Standardises with z-score
])

catPipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # Fills blank values with most frequent category
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)) # converts each category to a binary column
])