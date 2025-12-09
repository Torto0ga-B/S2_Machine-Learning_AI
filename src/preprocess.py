import pandas as pd
from sklearn.pipeline import Pipeline # Chains multiple transformations and a model together 
from sklearn.compose import ColumnTransformer # Allows for different preprocessing to different features
from sklearn.impute import SimpleImputer # Fills missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler # OHE converts categorical variables to numerical format for ML, SS does Z-score normalisation

def getPreprocessor(df: pd.DataFrame):
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges'] # list of numerical features 
    cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ] # list of categorical features

    numPipline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # If the numerical column has blank value, fill with mode
        ('scalar', StandardScaler()) # Standardises with z-score
    ])

    catPipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # Fills blank values with most frequent category
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)) # converts each category to a binary column
    ])

    # One unified object that handles all preprocessing when called and ensures it all occurs in the right order
    preprocessor = ColumnTransformer([
        ('num', numPipline, num_cols), # Applies the numeric pipeline to 'num_cols'
        ('cat', catPipeline, cat_cols) # Applies the categorical pipeline to 'cat_cols'
    ])

    return preprocessor, num_cols, cat_cols
