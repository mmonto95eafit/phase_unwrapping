from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Model
import os
import numpy as np


def build_model(input_layer, start_neurons):
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


# def image_generator(batchsize):
#     real_directory = 'data/train/real'
#     wrapped_directory = 'data/train/wrapped'
#     inputs = []
#     targets = []
#     batchcount = 0
#     while True:
#         for filename in os.listdir(real_directory):
#             real_file = f'{real_directory}/{filename}'
#             wrapped_file = f'{wrapped_directory}/{filename}'
#
#             img = load_img(real_file, color_mode='grayscale')
#             real_img = np.expand_dims(np.asarray(img) / 255, axis=2)
#
#             img = load_img(wrapped_file, color_mode='grayscale')
#             wrapped_img = np.expand_dims(np.asarray(img) / 255, axis=2)
#
#             inputs.append(real_img)
#             targets.append(wrapped_img)
#
#             batchcount += 1
#             if batchcount > batchsize:
#                 x = np.array(inputs, dtype='float32')
#                 y = np.array(targets, dtype='float32')
#                 yield x, y
#                 inputs = []
#                 targets = []
#                 batchcount = 0


def load_images():
    real_directory = 'data/train/real'
    wrapped_directory = 'data/train/wrapped'
    inputs = []
    targets = []
    for filename in os.listdir(real_directory):
        real_file = f'{real_directory}/{filename}'
        wrapped_file = f'{wrapped_directory}/{filename}'

        img = load_img(real_file, color_mode='grayscale')
        real_img = np.expand_dims(np.asarray(img) / 255, axis=2)

        img = load_img(wrapped_file, color_mode='grayscale')
        wrapped_img = np.expand_dims(np.asarray(img) / 255, axis=2)

        inputs.append(real_img)
        targets.append(wrapped_img)

    return np.array(inputs, dtype='float32'), np.array(targets, dtype='float32')


if __name__ == '__main__':
    batch_size = 100
    n_images = 4000

    x, y = load_images()

    input_layer = Input((240, 240, 1))
    output_layer = build_model(input_layer, 16)
    unet = Model(input_layer, output_layer)

    unet.compile(optimizer="adam", loss="mse")
    unet.fit(x, y, batch_size=batch_size, epochs=1)
