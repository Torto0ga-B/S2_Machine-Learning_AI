import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Load and prepare the data

df = pd.read_csv("data/Telco_Cusomer_Churn.csv") # Load csv into dataframe

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Converts the blank spaces to NaN

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1}) # Maps the Churn column to binary integers as ML models need numeric targets

x = df.drop("Churn", axis=1)
y = df["Churn"]

from src.preprocess import getPreprocessor
preprocessor, num_cols, cat_cols = getPreprocessor(df)

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

preprocessor.fit(x_train) # Fitted the preprocessor around the training set of data only, to avoid data leakage

# Transform the training, validation and test sets so that the categorical features are one-hot coded and the numeric features are z-score normalised
x_train_p = preprocessor.transform(x_train)
x_val_p = preprocessor.transform(x_val)
x_test_p = preprocessor.transform(x_test)


# Train the Logistic Regression Algorithm

log_reg = LogisticRegression(
    max_iter=1000, # Ensures convergence for large feature sets
    class_weight="balanced", # Due to class imbalance giving churners more weight
    solver="liblinear" # Works well at solving medium datasets with binary targets
)

log_reg.fit(x_train_p, y_train)


# Train the Random Forest Algorithm

RF = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

RF.fit(x_train_p, y_train)


# Hyperparameter tuning for RF
# Defines the grid of hyperparameters for RF
RF_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}
# GridSearchCV searches all combinaions using 3-fold cross validation
RF_grid = GridSearchCV(
    RF,
    RF_params,
    scoring = "recall", # Prioritise recall for churn prediction
    cv = 3,
    n_jobs = -1
)

RF_grid.fit(x_train_p, y_train)
best_RF = RF_grid.best_estimator_

print("best RF Params", RF_grid.best_params_)


# Evaluate the models using the validation set of data

def evaluate(model, xv, yv, name):
    pred = model.predict(xv)
    prob = model.predict_proba(xv)[:, 1] # Needed for the ROC-AUC

    print("=====", name, "=====")
    print("Accuracy:",  accuracy_score(yv, pred))
    print("Precision:", precision_score(yv, pred))
    print("Recall:",    recall_score(yv, pred))
    print("F1:",        f1_score(yv, pred))
    print("ROC-AUC:",   roc_auc_score(yv, prob))
    # Returns the metrics in a dictionary for an easy comparison
    return {
        "accuracy": accuracy_score(yv, pred),
        "precision": precision_score(yv, pred),
        "recall": recall_score(yv, pred),
        "f1": f1_score(yv, pred),
        "roc_auc": roc_auc_score(yv, prob)
    }

metrics_LR = evaluate(log_reg, x_val_p, y_val, "Logistic Regression")
metrics_RF = evaluate(best_RF, x_val_p, y_val, "Random Forest (Tuned)")

# Decision algorithm for best model
score_RF = 0
score_LR = 0

if metrics_RF["accuracy"] > metrics_LR["accuracy"]:
    score_RF = score_RF + 1
elif metrics_LR["accuracy"] > metrics_RF["accuracy"]:
    score_LR = score_LR + 1

if metrics_RF["precision"] > metrics_LR["precision"]:
    score_RF = score_RF + 1
elif metrics_LR["precision"] > metrics_RF["precision"]:
    score_LR = score_LR + 1

if metrics_RF["recall"] > metrics_LR["recall"]:
    score_RF = score_RF + 1
elif metrics_LR["recall"] > metrics_RF["recall"]:
    score_LR = score_LR + 1

if metrics_RF["f1"] > metrics_LR["f1"]:
    score_RF = score_RF + 1
elif metrics_LR["f1"] > metrics_RF["f1"]:
    score_LR = score_LR + 1

if metrics_RF["roc_auc"] > metrics_LR["roc_auc"]:
    score_RF = score_RF + 1
elif metrics_LR["roc_auc"] > metrics_RF["roc_auc"]:
    score_LR = score_LR + 1

if score_RF > score_LR:
    final_model = best_RF
    print("Selected Model: Random Forest")
else:
    final_model = log_reg
    print("Selected Model: Logistic Regression")

# Final Test Performance

print("=====FINAL TEST PERFORMANCE=====")
evaluate(final_model, x_test_p, y_test, "Final Model") # Applies the chosen model to the uneseen test set of data

# Generating SHAP explainations

# Uses TreeExplainer for random forest and LinearExplainer for Logistic Regression
explainer = shap.TreeExplainer(final_model) if isinstance(final_model, RandomForestClassifier) else shap.LinearExplainer(final_model, x_train_p)
shap_values = explainer.shap_values(x_test_p)

plt.figure()
shap.summary_plot(shap_values, x_test_p, show=False)
plt.savefig("results/shap_summary.png") # uses a sumary plot instead of displaying interactively for the paper
plt.close()

# Save the model and preprocessor
joblib.dump(preprocessor, "results/preprocessor.joblib")
joblib.dump(final_model, "results/final_model.joblib")
# Saves both the preprocessing pipeline and final model for reproduceability
print("Models saved successfully")
