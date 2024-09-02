import matplotlib.pyplot as plt
from scipy.constants import precision
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np

def evaluate_model(model, test_data_generator):
    # atleast to generate predictions and true labels too
    y_pred = model.predict(test_data_generator)
    y_true = test_data_generator.classes

    # then convert predictions to binary labels (let's assume it is binary classification)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)

    # time to calculate the metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)

    print(f"Accuracy: {accuracy:.4}")