import math

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


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


class PhaseUnwrappingSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size=32):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return (np.array([np.expand_dims(imread(file_name, as_gray=True) / 255., axis=2) for file_name in batch_x]),
                np.array([np.expand_dims(imread(file_name, as_gray=True) / 255., axis=2) for file_name in batch_y]))


def wrap_image(img):
    return np.angle(np.exp(1j * np.array(img)))


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
    real_directory = 'data/real_simple'
    wrapped_directory = 'data/wrapped_simple'
    inputs = []
    targets = []
    for filename in os.listdir(real_directory):
        real_file = f'{real_directory}/{filename}'
        wrapped_file = f'{wrapped_directory}/{filename}'

        img = load_img(real_file, color_mode='grayscale')
        real_img = np.expand_dims(np.asarray(img) / 255, axis=2)

        img = load_img(wrapped_file, color_mode='grayscale')
        wrapped_img = np.expand_dims(np.asarray(img) / 255, axis=2)

        targets.append(real_img)
        inputs.append(wrapped_img)

    return np.array(inputs, dtype='float32'), np.array(targets, dtype='float32')


if __name__ == '__main__':
    batch_size = 100
    n_images = 4000

    # x, y = load_images()
    real_directory = 'data/real_simple'
    wrapped_directory = 'data/wrapped_simple'
    wrapped_files = os.listdir(wrapped_directory)
    real_files = os.listdir(real_directory)

    x_set = [f'{wrapped_directory}/{file}' for file in wrapped_files if file in real_files]
    y_set = [f'{real_directory}/{file}' for file in wrapped_files if file in wrapped_files]

    sequence = PhaseUnwrappingSequence(x_set, y_set, batch_size=batch_size)

    input_layer = Input((128, 128, 1))
    # input_layer = Input((240, 240, 1))
    output_layer = build_model(input_layer, 16)
    unet = Model(input_layer, output_layer)

    unet.compile(optimizer="adam", loss="mse")

    # unet = load_model('data/models/128x128.h5')

    history = unet.fit(sequence, epochs=10)
    # history = unet.fit(x, y, batch_size=batch_size, epochs=10)

    unet.save(f'data/models/128x128.h5')

    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'])
    plt.savefig('data/plots/loss.png')

    for file in os.listdir(f'data/wrapped_simple')[:10]:
        img = load_img(f'data/wrapped_simple/{file}', color_mode='grayscale')
        wrapped_img = np.expand_dims(np.asarray(img) / 255, axis=2)
        predicted_img = unet.predict(np.array([wrapped_img]))
        predicted_img = predicted_img.reshape(predicted_img.shape[1], predicted_img.shape[2])

        img = load_img(f'data/real_simple/{file}', color_mode='grayscale')
        real_img = np.asarray(img) / 255

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(12, 8)
        ax1.imshow(wrapped_img.reshape(predicted_img.shape[0], predicted_img.shape[1]), cmap='gray')
        ax2.imshow(predicted_img, cmap='gray')
        ax3.imshow(real_img, cmap='gray')
        plt.savefig(f'data/plots/{file}')
