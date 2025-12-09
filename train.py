import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load and prepare the data

df = pd.read_csv("data/Telco_Customer_Churn.csv") # Load csv into dataframe
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1}) # Maps the Churn column to binary integers as ML models need numeric targets

x = df.drop("Churn", axis=1)
y = df["Churn"]



