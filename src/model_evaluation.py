import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np

def evaluate_model(model, test_data_generator):
    # atleast to generate predictions and true labels too
    y_pred = model.predict(test_data_generator)
    y_true = test_data_generator.classes

    # then convert predictions to binary labels (