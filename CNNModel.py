import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

X_train = X_train / 255
X_test = X_test / 255

cnn = models.Sequential(
    [layers.Conv2D(filters = 32, kernel_size = (9, 9), activation = 'relu', input_shape = (32, 32, 3), padding = 'same'),
     layers.MaxPooling2D((2, 2)),
     
     layers.Conv2D(filters = 48, kernel_size = (9, 9), activation = 'relu', padding = 'same'),
     layers.MaxPooling2D((2, 2)),
     
     layers.Flatten(), 
     layers.Dense(100, activation = 'relu'),
     layers.Dense(100, activation = 'relu'),
     layers.Dense(100, activation = 'relu'),
     layers.Dense(50, activation = 'relu'),
     layers.Dense(50, activation = 'relu'),
     layers.Dense(10, activation = 'softmax')])

cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

cnn.fit(X_train, y_train, epochs = 10)
