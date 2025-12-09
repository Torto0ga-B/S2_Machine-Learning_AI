import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocess import getPreprocessor, num_cols, cat_cols

# Load and prepare the data

df = pd.read_csv("data\Telco_Cusomer_Churn.csv") # Load csv into dataframe
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1}) # Maps the Churn column to binary integers as ML models need numeric targets

x = df.drop("Churn", axis=1)
y = df["Churn"]


# Train / validation / Test

# First split of training against the temp data

x_train, x_temp, y_train, y_temp = train_test_split(
    # 30% of the original dataset goes into the test size and you're left with 70%
    x, y, test_size = 0.30, stratify = y, random_state = 42
)

# Second split of data
x_val, x_test, y_val, y_test = train_test_split(
    # From that 30% it's halfed leaving 15% for each
    x_temp, y_temp, test_size = 0.50, stratify = y_temp, random_state = 42
)

print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)

# Preprocess all the data

getPreprocessor.fit(x_train) # Fitted the preprocessor around the training set of data only, to avoid data leakage

# Transform the training, validation and test sets so that the categorical features are one-hot coded and the numeric features are z-score normalised
x_train_p = getPreprocessor.transform(x_train)
x_val_p = getPreprocessor.transform(x_val)
x_test_p = getPreprocessor.transform(x_test)


# Train the Logistic Regression Algorithm

log_reg = LogisticRegression(
    max_iter=1000, # Ensures convergence for large feature sets
    class_weight="balanced", # Due to class imbalance giving churners more weight
    solver="liblinear" # Works well at solving medium datasets with binary targets
)

log_reg.fit(x_train_p, y_train)


# Train the Random Forest Algorithm

ran_for = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

ran_for.fit(x_train_p, y_train)


# Hyperparameter tuning for RF
# Defines the grid of hyperparameters for RF
ran_for_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_spilt": [2, 5, 10]
}
# GridSearchCV searches all combinaions using 3-fold cross validation
ran_for_grid = GridSearchCV(
    ran_for,
    ran_for_params,
    scoring = "recall", # Prioritise recall for churn prediction
    cv = 3,
    n_jobs = -1
)

ran_for_grid.fit(x_train_p, y_train)
best_ran_for = ran_for_grid.best_estimator_

print("best RF Params", ran_for_grid.best_params_)