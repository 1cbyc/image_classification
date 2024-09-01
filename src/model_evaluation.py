from tkinter import Image

import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from scipy.special import y_pred
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
    test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

    y_pred = model.predict(test_generator)
    y_pred = np.where