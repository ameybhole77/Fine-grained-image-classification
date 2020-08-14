# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:30:54 2018

"""
import gc
from statistics import mean

import numpy as np
import os
import time
import cv2
import skimage
from keras.models import load_model
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import resnet50
from resnet50 import ResNet50
# from vgg16 import VGG16
from delorean import now

from keras.applications.vgg16 import VGG16
from keras.models import model_from_json
from keras.preprocessing import image as Image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras_vggface import VGGFace
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.utils import np_utils, to_categorical
from sklearn import preprocessing
from keras import optimizers, regularizers, initializers
from keras.engine import Model
from keras.layers import Flatten, Dense, Input
# from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from keras.applications.imagenet_utils import preprocess_input

# from tensorflow.python.keras.models import load_model

import matplotlib.pyplot as plt


# import matplotlib.pyplot as plt
# from imagenet_utils import decode_predictions

def visualize_scatter_with_images(X_2d_data, images, figsize=(50,50), image_zoom=0.2):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()


def visualize_scatter(data_2d, label_ids, figsize=(25, 25)):
	plt.figure(figsize=figsize)
	plt.grid()

	nb_classes = len(np.unique(label_ids))
	index = 0
	for label_id in np.unique(label_ids):

		plt.scatter(data_2d[np.where(label_ids == label_id), 0],
					data_2d[np.where(label_ids == label_id), 1],
					marker='o',
					color=plt.cm.Set1(index / float(nb_classes)),
					linewidth='1',
					alpha=0.4,
					label= label_id
					)
		index += 1
	plt.legend(loc='best')
	plt.show()

def perform_tsne(images, labels):
	grayImages = []
	for image in images:
		gray = rgb2gray(image)
		gray = gray.flatten()
		grayImages.append(gray)
	print(grayImages[0].shape)

	pca = PCA(n_components=180)
	pca_result = pca.fit_transform(grayImages)
	print(pca_result.shape)
	tsne = TSNE(n_components=2, perplexity=40.0)
	tsne_result = tsne.fit_transform(pca_result)
	tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

	float_labels = [float (label) for label in labels]

	# VISUALIZATION OF TSNE !
	#visualize_scatter(tsne_result_scaled, float_labels)
	#visualize_scatter_with_images(tsne_result_scaled, images)

def write_result_image(image,pred_image, resultText, image_width):

	vertical_white = np.full([image_width, 40, 3], 255, dtype=np.uint8)
	horizontal_white = np.full([40, image_width*2 + 120, 3], 255, dtype=np.uint8)
	result_image = np.concatenate((vertical_white, image), axis=1)
	result_image = np.concatenate((result_image, vertical_white), axis=1)
	result_image = np.concatenate((result_image, pred_image), axis=1)
	result_image = np.concatenate((result_image, vertical_white), axis=1)
	result_image = np.concatenate((horizontal_white, result_image), axis=0)

	imageHeight, imageWidth, sceneNumChannels = result_image.shape
	SCALAR_Black = (0.0, 0.0, 0.0)

	fontFace = cv2.FONT_HERSHEY_TRIPLEX

	fontScale = 0.5
	fontThickness = 1

	fontThickness = int(fontThickness)

	upperLeftTextOriginX = int(imageWidth * 0.05)
	upperLeftTextOriginY = int(imageHeight * 0.05)

	textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
	textSizeWidth, textSizeHeight = textSize

	lowerLeftTextOriginX = upperLeftTextOriginX
	lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

	cv2.putText(result_image, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_Black, fontThickness)
	return result_image

def printWrongPredictions(pred, y_test):
	array = []
	image_width = 0
	if training_model == "FACENET":
		image_width = 160
	else:
		image_width = 224

	wrong_predictions = np.full([1, image_width*2 + 120, 3], 255, dtype=np.uint8)

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
	print(y_test)

	for i in range(len(y_test)):
		if y_test[i] != pred[i]:
			print("Predicted:", pred[i], " Actual: ", y_test[i])
			val = np.where(y_test == pred[i])[0][0]
			result_text = "Predicted:"+ pred[i]+ " Actual: "+ y_test[i]
			vis1 = write_result_image(X_test[val], X_test[i],result_text, image_width)
			wrong_predictions = np.concatenate((wrong_predictions, vis1), axis=0)
	#cv2.imshow('wrong-predictions.png', wrong_predictions)

def plot_data_graph(hist):
	import matplotlib.pyplot as plt
	# visualizing losses and accuracy
	train_loss = hist.history['loss']
	val_loss = hist.history['val_loss']
	train_acc = hist.history['categorical_accuracy']
	val_acc = hist.history['val_categorical_accuracy']
	xc = range(nb_epochs)

	plt.figure(1, figsize=(12, 10))
	plt.plot(xc, train_loss)
	plt.plot(xc, val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.grid(True)
	plt.legend(['train', 'val'])
	plt.show()

	plt.figure(2, figsize=(12, 10))
	plt.plot(xc, train_acc)
	plt.plot(xc, val_acc)
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.grid(True)
	plt.legend(['train', 'val'], loc=4)
	plt.show()


def plot_hist(hist_array):
	xc = range(nb_epochs)

	loss_array = []
	val_loss_array =[]
	accuracy_array = []
	val_accuracy_array = []

	for model_histories in hist_array:
		loss_array.append(model_histories.history['loss'])
		val_loss_array.append(model_histories.history['val_loss'])
		accuracy_array.append(model_histories.history['categorical_accuracy'])
		val_accuracy_array.append(model_histories.history['val_categorical_accuracy'])

	print()

	plt.figure(1)
	plt.plot(xc, model_histories.history['loss'])
	plt.plot(xc, model_histories.history['val_loss'])
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.show()

	plt.figure(2)
	plt.plot(xc, model_histories.history['categorical_accuracy'], labels="Accuracy")
	plt.plot(xc, val_accuracy_array, labels="Accuracy")
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.legend()
	plt.show()

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

nb_classes = 136
nb_epochs = 10
n_split = 5
b_size = 2
augment_data = False
PATH = os.getcwd()
data_path = PATH + '/Dataset_resized'
data_dir_list = os.listdir(data_path)
plot_data = True
training_model = "VGG-16"

img_data_list = []
labels = []
list_of_image_paths = []
label_test = []

for folder_name in data_dir_list:
	img_list = os.listdir(data_path + '/' + folder_name)
	for image in img_list:
		retrieve_dir = data_path + "/" + folder_name + "/" + image
		images = cv2.imread(retrieve_dir, 3)
		if training_model == "FACENET":
			images = cv2.resize(images, (160, 160))  # FOR FACENET RESIZE
		list_of_image_paths.append(images)
	img_data_list.append(img_list)
	labels.append(folder_name)
	label_test.append([folder_name] * len(img_list))

#perform_tsne(list_of_image_paths, labels)

list_of_image_paths = np.array(list_of_image_paths)
flattened_list = np.asarray([y for x in label_test for y in x], dtype="str")
lb = preprocessing.LabelBinarizer().fit(flattened_list)
labels = lb.transform(flattened_list)
print("Number of Labels: ", len(labels))

skf = StratifiedShuffleSplit(n_splits=n_split, random_state=None, test_size=0.2)
print(skf.n_splits)

all_fold_accuracy = []
all_fold_loss = []
counter = 1

for train_index, test_index in skf.split(list_of_image_paths, labels):
	X_train, X_test = list_of_image_paths[train_index], list_of_image_paths[test_index]
	y_train, y_test = labels[train_index], labels[test_index]
	print(y_train.shape)
	print(type(y_train))
	augmented_Xtrain = []
	y_train_labels = []

	images = []
	#train_datagen = ImageDataGenerator(rotation_range=10,horizontal_flip=True, vertical_flip=True, zoom_range=0.4)
	if augment_data:
		for index in range(len(X_train)):

			image = X_train[index]
			label = y_train[index]

			# plt.imshow((image).astype(np.uint8))
			# 		# plt.title("Original")
			# 		# plt.show()
			# 		# time.sleep(10)

			flipped_image = flip_images(image)
			# plt.imshow((flipped_image).astype(np.uint8))
			# plt.title("Flipped Image")
			# plt.show()
			# time.sleep(10)

			rotatedMinus10Image = rotate(image, 10)
			# plt.imshow((rotatedMinus10Image).astype(np.uint8))
			# plt.title("Rotated Minus 10 Degrees")
			# plt.show()
			# time.sleep(10)

			rotatedPlus10Image = rotate(image,  -10)
			# plt.imshow((rotatedPlus10Image).astype(np.uint8))
			# plt.title("Rotated Plus 10 Degrees")
			# plt.show()
			# time.sleep(10)

			gaussianImage = gaussian_noise(image)
			# plt.imshow((gaussianImage).astype(np.uint8))
			# plt.title("Gaussian Noise")
			# plt.show()
			# time.sleep(10)

			darkerImage = lighting(image, 0.5)
			# plt.imshow((darkerImage).astype(np.uint8))
			# plt.title("Darkened Image")
			# plt.show()
			# time.sleep(10)

			lighterImage = lighting(image,  1.5)
			# plt.imshow((lighterImage).astype(np.uint8))
			# plt.title("Lighter Image")
			# plt.show()
			# time.sleep(10)

			augmented_Xtrain.extend([image, flipped_image, rotatedMinus10Image, rotatedPlus10Image, gaussianImage, darkerImage, lighterImage])
			y_train_labels.extend([label]*7)

			X_train = np.uint8(augmented_Xtrain)
			y_train = np.int32(y_train_labels)
	else:
		X_train = X_train
		y_train = y_train

	# training_model 1 for VGGFace 2 for VGG16
	im_shape = (224, 224, 3)
	image_input = Input(shape=im_shape)

	##############################################################################
	# RESNET 50
	##############################################################################
	if training_model == "ResNet50":
		base_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
		x = base_resnet.layers[-1]
		out = Dense(units=nb_classes, activation='softmax', name='output', use_bias=True,
					kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01),
					kernel_regularizer=regularizers.l2(0.001),
					bias_initializer='zeros', bias_regularizer=regularizers.l2(0.001)
					)(x.output)
		custom_resnet_model = Model(inputs=base_resnet.input, outputs=out)

		for layer in custom_resnet_model.layers:
			layer.trainable = True
		#custom_resnet_model.summary()
		# sgd = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)
		# adadelta = optimizers.Adadelta(lr=1e-2)
		# nadam = optimizers.nadam(lr=1)
		adam = optimizers.adam(lr=0.0001)
		custom_resnet_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

		t = time.time()
		hist = custom_resnet_model.fit(X_train, y_train, batch_size=b_size, epochs=nb_epochs, verbose=1,
									   validation_data=(X_test, y_test))
		print('Training time: %s' % (t - time.time()))
		(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=b_size, verbose=1)

		print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
		pred = custom_resnet_model.predict(X_test)
		y_test_labels = lb.inverse_transform(y_test)
		printWrongPredictions(pred, y_test)
		pred = lb.inverse_transform(pred)

		if plot_data:
			plot_data_graph(hist)

		all_fold_accuracy.append(accuracy * 100)
		all_fold_loss.append(loss)

		# CM
		cm = metrics.confusion_matrix(y_test_labels, pred)
		cas = plt.imshow(cm, cmap='Greys', interpolation='nearest')
		plt.xlabel("Predicted labels")
		plt.ylabel("True labels")
		plt.title('Confusion matrix')
		plt.colorbar()
		plt.clim(0, 1);
		plt.show()

		setOfLabels = list(set(y_test_labels.flatten()))
		print(classification_report(y_test_labels, pred, target_names=setOfLabels))
		counter = counter + 1

	####################################################################
	# FACENET#
	####################################################################
	if training_model == "FACENET":
		facenet_model = load_model('facenet_keras.h5')
		weights = facenet_model.load_weights('facenet_keras_weights.h5')
		x = facenet_model.layers[-3]
		denseLayer = Dense(nb_classes, activation='softmax', name='output', use_bias=True,
						   #             kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01),
						   #             kernel_regularizer=regularizers.l2(0.001),
						   #             bias_initializer='zeros', bias_regularizer=regularizers.l2(0.001))(x.output)
						   )(x.output)
		custom_facenet_model = Model(inputs=facenet_model.input, outputs=denseLayer)
		#custom_facenet_model.summary()
		adam = optimizers.adam(lr=0.0001)
		custom_facenet_model.compile(loss='categorical_crossentropy', optimizer=adam,
									 metrics=['categorical_accuracy'])
		hist = custom_facenet_model.fit(X_train, y_train, batch_size=b_size, epochs=nb_epochs, verbose=1,
								 validation_data=(X_test, y_test))
		(loss, accuracy) = custom_facenet_model.evaluate(X_test, y_test, batch_size=b_size, verbose=1)
		print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
		pred = custom_facenet_model.predict(X_test)
		printWrongPredictions(pred, y_test)

		y_test_labels = lb.inverse_transform(y_test)
		pred = lb.inverse_transform(pred)

		if plot_data:
			plot_data_graph(hist)

		all_fold_accuracy.append(accuracy * 100)
		all_fold_loss.append(loss)

		cm = metrics.confusion_matrix(y_test_labels, pred)
		cas = plt.imshow(cm, cmap='Greys', interpolation='nearest')
		plt.xlabel("Predicted labels")
		plt.ylabel("True labels")
		plt.title('Confusion matrix')
		plt.colorbar()
		plt.clim(0, 1);
		plt.show()

		setOfLabels = list(set(y_test_labels.flatten()))
		print(classification_report(y_test_labels, pred, target_names=setOfLabels))
		counter = counter + 1

# ################VGGFace Model############
# if training_model == 1:
#
# 	hidden_dim = 2048
#
# 	vggface_model = VGGFace(input_tensor=image_input, include_top=True, weights='vggface', input_shape=im_shape,
# 		                     pooling='max')
# 	vggface_model.summary()
#
# 	pool5 = vggface_model.layers[-8]
#
# 	flatten = vggface_model.layers[-7]
#
# 	fc6 = vggface_model.layers[-6]
# 	fc6relu = vggface_model.layers[-5]
#
# 	fc7 = vggface_model.layers[-4]
# 	fc7relu = vggface_model.layers[-3]
#
# 	fc8 = vggface_model.layers[-2]
# 	fc8relu = vggface_model.layers[-1]
#
# 	dropout1 = Dropout(0.5)
# 	dropout2 = Dropout(0.5)
#
# 	# Reconnect the layers
# 	x = dropout1(pool5.output)
# 	x = flatten(x)
# 	x = fc6(x)
# 	x = fc6relu(x)
# 	x = dropout2(x)
# 	x = fc7(x)
# 	x = fc7relu(x)
# 	out = Dense(nb_classes, activation='softmax', name='output')(x)
#
# 	custom_vggface_model = Model(vggface_model.input, output=out)
# 	 # Resnet_model = ResNet50(weights='imagenet',include_top=False)
#
# 	vggface_model.summary()
# 	custom_vggface_model.summary()
#
# 	for layer in custom_vggface_model.layers[:-1]:
# 		 layer.trainable = False
#
# 	custom_vggface_model.layers[-1].trainable
# 	custom_vggface_model.summary()
#
# 	sgd = optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
# 	custom_vggface_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
# 	t = time.time()
# 	hist = custom_vggface_model.fit(augmented_X_train, y_train_labels, batch_size= b_size, epochs=nb_epochs, verbose=1,
# 		                             validation_data=(X_test, y_test))
# 	print('Training time: %s' % (t - time.time()))
# 	(loss, accuracy) = custom_vggface_model.evaluate(X_test, y_test, batch_size= b_size, verbose=1)
#
# 	print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
#
	# #################VGG Model############
	if training_model == "VGG-16":
		vgg_model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
		vgg_model.summary()
		last_layer = vgg_model.get_layer('fc2').output
		##x= Flatten(name='flatten')(last_layer)
		out = Dense(nb_classes, activation='softmax', name='output')(last_layer)
		custom_vgg_model = Model(image_input, out)
		custom_vgg_model.summary()

		for layer in custom_vgg_model.layers:
			layer.trainable = True

		custom_vgg_model.summary()
		adam = optimizers.adam(lr=0.0001)
		sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
		custom_vgg_model.compile(loss='categorical_crossentropy', optimizer=adam ,metrics=['categorical_accuracy'])

		t=time.time()

		hist = custom_vgg_model.fit(X_train, y_train, batch_size = b_size, epochs = nb_epochs, verbose=1, validation_data=(X_test, y_test))
		print('Training time: %s' % (t - time.time()))
		(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=b_size, verbose=1)

		print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


		pred = custom_vgg_model.predict(X_test)
		printWrongPredictions(pred, y_test)

		y_test_labels = lb.inverse_transform(y_test)
		pred = lb.inverse_transform(pred)

		if plot_data:
			plot_data_graph(hist)

		all_fold_accuracy.append(accuracy * 100)
		all_fold_loss.append(loss)

		cm = metrics.confusion_matrix(y_test_labels, pred)
		cas = plt.imshow(cm, cmap='Greys', interpolation='nearest')
		plt.xlabel("Predicted labels")
		plt.ylabel("True labels")
		plt.title('Confusion matrix')
		plt.colorbar()
		plt.clim(0, 1);
		plt.show()

		setOfLabels = list(set(y_test_labels.flatten()))
		print(classification_report(y_test_labels, pred, target_names=setOfLabels))
		counter = counter + 1


# PRINT OVERALL ACCURACY AND MEAN OVER ALL FOLDS
print("Mean accuracy over all folds: ", mean(all_fold_accuracy))
print("Mean loss over all folds: ", mean(all_fold_loss))















