# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 16:27:27 2021

@author: ahasi
"""
#Imports
import os
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical, plot_model
import cv2

#Defining the constants
datapath = os.path.join("./Dataset/")
X, y= [],[]
img_rows=200
img_cols=200

#Load the files into the variables X & y
def loadFiles(X, y, path):
    for image in os.listdir(path):
        img=cv2.imread(datapath+image, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, (img_rows, img_cols))
        X.append(img)
        y.append(int(image[0]))
    X=np.asarray(X)
    y=np.asarray(y)
    return X,y

X, y = loadFiles(X, y, datapath)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=110)

X_train = X_train.reshape(5685, 200, 200, 1)
X_test = X_test.reshape(2801, 200, 200, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = to_categorical(y_train, num_classes = 10 )
y_test = to_categorical(y_test, num_classes= 10)

model = Sequential([
    keras.layers.Convolution2D(64, 3, activation='relu', input_shape=(200,200,1), strides=1),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.1),
    keras.layers.Convolution2D(64, 3, activation='relu', strides=1),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.1),
    keras.layers.Convolution2D(128, 3, activation='relu', strides=1),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Convolution2D(32, 3, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    ])


model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.summary()
data_gen=ImageDataGenerator(rotation_range=45, horizontal_flip=True, rescale=0.7, validation_split=0.2)
data_gen.fit(X_train)

model.fit_generator(data_gen.flow(X_train, y_train, batch_size=32),
          steps_per_epoch=32, epochs=800, 
          validation_data=(X_test, y_test))

score=model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", score[0])
print("Test accuracy", score[1])

model.save("model70")



