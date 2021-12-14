from tensorflow.keras import callbacks, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

IMG_DIM = (48, 48)
CLASSES = os.listdir(os.path.join(os.curdir, 'train'))
N_CLASSES = len(CLASSES)

def load_images(directory, shuffle=False):
    X = []
    y = []

    for i, klass in enumerate(CLASSES):
        for _, _, filenames in os.walk(f'{directory}/{klass}'):
            images = [tf.io.decode_jpeg(tf.io.read_file(f'{directory}/{klass}/{fname}'), channels=3)
                      for fname in filenames]
            X.extend(images)
            y.extend([i] * len(images))

    idx = np.arange(len(X))

    if shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(idx)

    return np.array(X)[idx], np.array(y)[idx]


def build_conv2d_model(n_convd_layers=1, n_dense_layers=1, n_neurons=50):
    model = models.Sequential([
        layers.InputLayer(input_shape=IMG_DIM + (3,)),
        layers.Rescaling(1./255)
    ])

    for i in range(n_convd_layers):
        model.add(layers.Conv2D(filters=32 * (i + 1), kernel_size=3, activation='relu'))
        model.add(layers.MaxPool2D(pool_size=2))

    model.add(layers.Flatten())

    for _ in range(n_dense_layers):
        model.add(layers.Dense(n_neurons, activation='relu'))

    model.add(layers.Dense(N_CLASSES, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer='rmsprop'
    )

    return model


def make_callbacks():
    return [callbacks.EarlyStopping(monitor='val_loss', patience=5)]


def build_and_train_model(trainX, trainy, n_convd_layers, n_dense_layers, n_neurons):
    model = build_conv2d_model(n_convd_layers=n_convd_layers,
                               n_dense_layers=n_dense_layers,
                               n_neurons=n_neurons)

    model.fit(trainX, trainy,
            epochs=50,
            validation_split=.2,
            batch_size=32,
            callbacks=make_callbacks())

    return model


if __name__ == '__main__':
    trainX, trainy = load_images('train')

    model = build_and_train_model(trainX, trainy, 2, 2, 50)
    model.save('model.h5')
