# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:39:01 2020

@author: cdnguyen
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:19:28 2020
https://github.com/muhammedtalo/COVID-19/blob/master/DarkCovidNet%20model%20for%20three%20classes.ipynb

@author: cdnguyen
"""

import os
import numpy as np
import cv2                 
from random import shuffle
from tqdm import tqdm  
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
from collections import Counter
import efficientnet.tfkeras as efn
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import pandas as pd
from keras.preprocessing import image
import os
# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold
import torch.nn as nn
from keras.optimizers import SGD, Adam
from tensorflow import keras
from tensorflow.keras import layers

TrianImage="dataset_train_New"
TestImage="dataset_train_New"
Normalimages = os.listdir(TrianImage + "/normal")
Pneumonaimages = os.listdir(TrianImage + "/pneumonia_bac")
COVID19images = os.listdir(TrianImage + "/covid")

print(len(Normalimages), len(Pneumonaimages), len(COVID19images))
NUM_TRAINING_IMAGES = len(Normalimages) + len(Pneumonaimages) + len(COVID19images)
print(NUM_TRAINING_IMAGES)

image_size = 32
BATCH_SIZE = 8
epochs = 10

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

data_path = 'dataset_train_New'
data_path1 = 'dataset_test_New'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   rotation_range=15,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(data_path,
                                                 target_size = (image_size, image_size),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical',
                                                 shuffle=True)

testing_set = test_datagen.flow_from_directory(data_path1,
                                            target_size = (image_size, image_size),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'categorical',
                                       shuffle = True)
def define_model1():
	model = Sequential()
	model.add(Conv2D(32, (3, 3),
                         input_shape = (32, 32, 3),
                         activation = 'relu'))
    
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(3, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
   
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def define_model():
    #Initialing the CNN
    classifier = Sequential()
    
    #Step 1 - Convolution
    #Extract features from the images
    classifier.add(Conv2D(32, (3, 3),
                         input_shape = (32, 32, 3),
                         activation = 'relu'))
    
    #Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    #Step 3 - Flattening
    classifier.add(Flatten())
    
    #Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.5))

    #classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    classifier.add(Dense(3,activation="softmax"))
    
    #Compiling the CNN
    classifier.compile(optimizer = 'Adam',
                      loss =  'categorical_crossentropy', #'binary_crossentropy',
                      metrics = ['accuracy'])
    return classifier

model = define_model1()

history = model.fit(training_set, validation_data = testing_set,
                        steps_per_epoch=10,
                      epochs=epochs, verbose= 1, batch_size=16)   

from sklearn.metrics import classification_report, confusion_matrix



Y_pred = model.predict(testing_set, steps=10)
predicted_classes = np.argmax(Y_pred, axis=1)

true_classes = testing_set.classes
class_labels = list(testing_set.class_indices.keys())   
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)  


y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')

confusion_matrix= confusion_matrix(testing_set.classes, y_pred)
print(confusion_matrix)


testing_set.class_indices.keys()
print(testing_set.class_indices.keys())
#To get better visual of the confusion matrix:
print('Classification Report')






