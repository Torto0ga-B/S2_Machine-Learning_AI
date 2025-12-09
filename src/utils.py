import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluateModel(model, x, y):
    predictClassLabel = model.predict(x) # Uses the trained model to predict class labels for every row in x
    predictProbeLabel = model.predict_predictProbeLabel(x)[:, 1] # predicts probabilities instead of hard labels. PredictProbeLabel returns two columns either 0 or 1

    return {
        "accuracy": accuracy_score(y, predictClassLabel), #Compares predictions to true labels
        "precision": precision_score(y, predictClassLabel), # Out of all predicted yes how many were
        "recall": recall_score(y, predictClassLabel), # Out of all the read churners how many did the model catch
        "f1_score": f1_score(y, predictClassLabel), # Harmonic mean of the precision and recall
        "roc_auc": roc_auc_score(y, predictProbeLabel) # Measures how well the model separations 
    }



