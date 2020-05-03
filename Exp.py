# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:30:54 2018

"""
import random

import keras
import skimage
from collections import Counter
import numpy as np
import os

import time

import cv2
import math
import tensorflow as tf
from keras.applications.resnet50 import ResNet50

import resnet50
from keras.models import load_model
from resnet50 import ResNet50
#from vgg16 import VGG16
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.utils import np_utils, to_categorical
from sklearn import preprocessing
from keras import optimizers, regularizers, initializers, Sequential
from keras.engine import Model
from bias import custom_layer

from keras.layers import Flatten, Dense, Input

PATH = os.getcwd()
data_path = PATH + '/Dataset_resized'
data_dir_list = os.listdir(data_path)

img_data_list = []
labels = []
folderCounter = 0

def flip_images(image):
    flipped_image = np.fliplr(image)
    return flipped_image

def rotate(image, angle):
    rotated_image = skimage.transform.rotate(image, angle= angle, preserve_range=True).astype(np.uint8)
    return rotated_image

def gaussian_noise(image):
    mean = 20;
    std = 20;
    noisy_img = image + np.random.normal(mean, std, image.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    return noisy_img_clipped

def lighting(image, gamma):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   flippedLighting = cv2.LUT(image, table)
   return flippedLighting

list_of_image_paths = []
labelTest = []

for eachfolderName in data_dir_list:
    img_list = os.listdir(data_path + '/' + eachfolderName)

    for image in img_list:
        retrieve_dir = data_path + "\\" + eachfolderName + "\\" + image
        images = cv2.imread(retrieve_dir, 3)
        images = cv2.resize(images, (160, 160)) #FOR FACENET RESIZE
        list_of_image_paths.append(images)
    labels.append(eachfolderName)
    labelTest.append([eachfolderName] * len(img_list))
    imageCounter = 0
    folderCounter = folderCounter + 1;

list_of_image_paths = np.array(list_of_image_paths)
flattened_list = np.asarray([y for x in labelTest for y in x], dtype="str")
lb = preprocessing.LabelBinarizer()
lb.fit(flattened_list)
list_of_image_paths_processed = resnet50.preprocess_input(list_of_image_paths)
X_train, X_test, y_train, y_test = train_test_split(list_of_image_paths_processed, flattened_list, test_size=0.2, stratify= flattened_list, shuffle=True)

y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

# augmented_Xtrain = []
# y_train_labels = []
# for index in range(len(X_train)):
#
#     image = X_train[index]
#     label = y_train[index]
#
#     flipped_image = flip_images(image)
#     rotatedMinus10Image = rotate(image, 10)
#     rotatedPlus10Image = rotate(image,  -10)
#     gaussianImage = gaussian_noise(image)
#     darkerImage = lighting(image, 0.5)
#     lighterImage = lighting(image,  1.5)
#     augmented_Xtrain.extend([image, flipped_image, rotatedMinus10Image, rotatedPlus10Image, gaussianImage, darkerImage, lighterImage])
#     y_train_labels.extend([label]*7)
#
# print(len(augmented_Xtrain))
# print(len(y_train_labels))

#############################Resnet50#############################################
# base_resnet = ResNet50(weights ='imagenet', include_top=False, pooling= 'avg')
# x = base_resnet.layers[-1]
# out = Dense(units=136, activation='softmax', name='output',use_bias=True,
#             kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01),
#             kernel_regularizer=regularizers.l2(0.001),
#             bias_initializer='zeros', bias_regularizer=regularizers.l2(0.001)
#             )(x.output)
# custom_resnet_model = Model(inputs = base_resnet.input, outputs=out)
# for layer in custom_resnet_model.layers:
#     layer.trainable = True
# custom_resnet_model.summary()
# sgd = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)
# adadelta = optimizers.Adadelta(lr=1e-2)
# nadam = optimizers.nadam(lr=1)
# adam = optimizers.adam(lr=0.0001)
# custom_resnet_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
#
# t = time.time()
# hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1,
#                                 validation_data=(X_test, y_test))
# print('Training time: %s' % (t - time.time()))
# (loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=32, verbose=1)
#
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
# pred = custom_resnet_model.predict(X_test)
# y_test_labels = lb.inverse_transform(y_test)
# y_classes = []
# array = []


facenet_model = load_model('facenet_keras.h5')
weights = facenet_model.load_weights('facenet_keras_weights.h5')
x = facenet_model.layers[-3]
denseLayer = Dense(136,  activation='softmax', name='output',use_bias=True,
#             kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01),
#             kernel_regularizer=regularizers.l2(0.001),
#             bias_initializer='zeros', bias_regularizer=regularizers.l2(0.001))(x.output)
            )(x.output)
# batchNorm =
custom_facenet_model = Model(inputs = facenet_model.input, outputs=denseLayer)
custom_facenet_model.summary()

custom_facenet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
custom_facenet_model.fit(X_train, y_train, batch_size = 32, epochs= 10, verbose=1, validation_data=(X_test, y_test))
(loss, accuracy) = custom_facenet_model.evaluate(X_test, y_test, batch_size=32, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
pred = custom_facenet_model.predict(X_test)


array = []
# Print predicted vs. Actual
for i in range(len(pred)):
    biggest_value_index = pred[i].argmax(axis=0)
    value = pred[i][biggest_value_index]
    y_classes = pred[i] >= value
    y_classes = y_classes.astype(int)
    array.append(y_classes)
predicted_list = np.asarray(array, dtype="int32")

y_test = lb.inverse_transform(y_test)
pred = lb.inverse_transform(predicted_list)

for i in range(len(y_test)):
    if y_test[i] != pred[i]:
        print("Predicted:", pred[i]," Actual: ", y_test[i])
