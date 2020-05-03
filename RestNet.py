import numpy as np
from resnet50 import ResNet50
from keras.preprocessing import image
from keras_applications import resnet50

def main():
    ResNet50_model()

def ResNet50_model():
    # Custom_resnet_model_1
    # Training the classifier alone
    image_input = Input(shape=(224, 224, 3))

    model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
    model.summary()
    last_layer = model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(num_classes, activation='softmax', name='output_layer')(x)
    custom_resnet_model = Model(inputs=image_input, outputs=out)
    custom_resnet_model.summary()

    for layer in custom_resnet_model.layers[:-1]:
        layer.trainable = False

    custom_resnet_model.layers[-1].trainable

    custom_resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    t = time.time()
    hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1,
                                   validation_data=(X_test, y_test))
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))