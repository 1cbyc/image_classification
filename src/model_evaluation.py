from tkinter import Image

import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import  classification_report, confusion_matrix

def plot_metrics(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evaluate_model(model, test_data_dir):
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator