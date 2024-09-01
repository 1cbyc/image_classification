import matplotlib.pyplot as plt
from sklearn.metrics import  classification_report, confusion_matrix

def plot_metrics(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evaluate_model(model, test_data_dir):
    test_datagen 