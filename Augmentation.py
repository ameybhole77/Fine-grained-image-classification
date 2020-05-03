import time

import cv2
import lasso as lasso
import numpy as np
import os
import skimage.transform
from keras import Input, Model
from keras.layers import Flatten, Dense
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_validate
from resnet50 import ResNet50


def main():
    getImageFileNames()


def getImageFileNames():
    PATH = os.getcwd()
    data_path = PATH + '/Dataset_resized'
    data_dir_list = os.listdir(data_path)

    img_data_list = []
    labels = []
    folderCounter = 0

    list_of_image_paths = []
    labelTest = []
    for eachfolderName in data_dir_list:
        #print(eachfolderName)
        img_list = os.listdir(data_path + '/' + eachfolderName)

        for image in img_list:
            retrieve_dir = data_path + "\\" + eachfolderName+ "\\" + image
            images = cv2.imread(retrieve_dir, 3)
            list_of_image_paths.append(images)

        img_data_list.append(img_list)
        labels.append(eachfolderName)

        labelTest.append([eachfolderName]*len(img_list))
        imageCounter = 0
        # print(img_list)
        # print(labelTest)

        # XTrain, XTest, YTrain, YTest = train_test_split(img_list, labelTest, test_size=0.2, random_state=42,
        #         #                                                shuffle=True)
        #
        #         # for eachImage in img_list:
        #         #     print(img_data_list[folderCounter][imageCounter])
        #         #     retrieve_dir = data_path + "\\" + labels[folderCounter] + "\\" + img_data_list[folderCounter][imageCounter]
        #         #     image = cv2.imread(retrieve_dir)
        #         #     flip_images(image, data_path, labels, folderCounter, img_data_list, imageCounter)
        #         #     rotate(image, data_path, labels, folderCounter, img_data_list, imageCounter, 10, 'rotate_minus_10')
        #         #     rotate(image, data_path, labels, folderCounter, img_data_list, imageCounter, -10, 'rotate_plus_10')
        #         #     gaussian_noise(image, data_path, labels, folderCounter, img_data_list, imageCounter)
        #         #     lighting(image, data_path, labels, folderCounter, img_data_list, imageCounter, 0.5, 'Darker')
        #         #     lighting(image, data_path, labels, folderCounter, img_data_list, imageCounter, 1.5, 'Lighter')
        #         #     imageCounter = imageCounter + 1;
        folderCounter = folderCounter + 1;
    #print(list_of_image_paths)

    list_of_image_paths = np.array(list_of_image_paths)
    # print(list_of_image_paths.shape)
    flattened_list = [y for x in labelTest for y in x]
    flattened_list = np.array(flattened_list, dtype='str')
    # print(flattened_list)
    lb = preprocessing.LabelBinarizer()
    lb.fit(flattened_list)
    print(lb.classes_)
    result = lb.transform(flattened_list)
    print(result.shape)

    X_train, X_test, y_train, y_test = train_test_split(list_of_image_paths, flattened_list, test_size=0.2, random_state=2)

    # Custom_resnet_model_1
    # Training the classifier alone
    image_input = Input(shape=(224, 224, 3))

    model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
    #model.summary()
    last_layer = model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(1, activation='softmax', name='output_layer')(x)
    custom_resnet_model = Model(inputs=image_input, outputs=out)
    custom_resnet_model.summary()

    for layer in custom_resnet_model.layers[:-1]:
        layer.trainable = False

    custom_resnet_model.layers[-1].trainable

    custom_resnet_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    t = time.time()
    hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1,
                                   validation_data=(X_test, y_test))
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    print('Training time: %s' % (t - time.time()))

 #cv_results = cross_validate(model, list_of_image_paths, flattened_list, cv=5, return_train_score = True)

def flip_images(image, data_path, labels, folderCounter, img_data_list, imageCounter):
    save_dir_flipped = data_path + "\\" + labels[folderCounter] + "\\" + 'flipped' + img_data_list[folderCounter][
        imageCounter]
    flipped_image = np.fliplr(image)
    cv2.imwrite(os.path.join(save_dir_flipped), flipped_image)


def rotate(image, data_path, labels, folderCounter, img_data_list, imageCounter, angle, dir_name):
    save_dir_rotate = data_path + "\\" + labels[folderCounter] + "\\" + dir_name + img_data_list[folderCounter][
        imageCounter]
    rotated_image = skimage.transform.rotate(image, angle= angle, preserve_range=True).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir_rotate), rotated_image)


def gaussian_noise(image, data_path, labels, folderCounter, img_data_list, imageCounter):
    save_dir_flipped = data_path + "\\" + labels[folderCounter] + "\\" + 'gaussian_noise_' + img_data_list[folderCounter][
        imageCounter]
    mean = 20;
    std = 20;
    noisy_img = image + np.random.normal(mean, std, image.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    cv2.imwrite(os.path.join(save_dir_flipped), noisy_img_clipped)


def lighting(image, data_path, labels, folderCounter, img_data_list, imageCounter, gamma, darkOrLight):
   save_dir_lighting= data_path + "\\" + labels[folderCounter] + "\\" + darkOrLight + img_data_list[folderCounter][imageCounter]
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   flippedLighting = cv2.LUT(image, table)
   cv2.imwrite(os.path.join(save_dir_lighting), flippedLighting)


if __name__ == '__main__':
    main()
