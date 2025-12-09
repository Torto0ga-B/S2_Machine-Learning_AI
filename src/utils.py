import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluateModel(model, x, y):
    predictClassLabel = model.predict(x) # Uses the trained model to predict class labels for every row in x
    predictProbLabel = model.predict_predictProbeLabel(x)[:, 1] # predicts probabilities instead of hard labels. PredictProbeLabel returns two columns either 0 or 1
    

