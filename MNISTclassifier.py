import argparse
import sys
import tempfile
import matplotlib.pyplot as plt
import time
import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, SimpleRNN, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import initializers
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D,UpSampling2D, Reshape
from keras import regularizers


def liniar_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Dense(512, activation='sigmoid', input_shape=x_train.shape[1:]))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
              batch_size=32,
              epochs=20,
              verbose=1,
              validation_data=(x_test, y_test))

    plotter(hist)


def mlp(lr):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()

    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    hist=model.fit(x_train, y_train,
              epochs=20,
              batch_size=128,
                   verbose=2,
                   validation_data=(x_train,y_train))

    b=model.fit(x_test, y_test,
                   epochs=20,
                   batch_size=128)
    return hist, b


def plot_mlp(hist, b):
    plt.plot(b.history['loss'])
    plt.plot(hist.history['loss'])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("model loss")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(b.history['acc'])
    plt.plot(hist.history['acc'])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("model accuracy")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def convent():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)


    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
              batch_size=128,
              epochs=20,
              verbose=1,
              validation_data=(x_test, y_test))
    plotter(hist)


def plotter(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("model loss")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("model accuracy")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def HyperParameterExploration():
    hist1, b1 = mlp(0.001)
    hist2, b2 = mlp(0.005)
    hist3, b3 = mlp(0.01)
    hist4, b4 = mlp(0.05)
    hist5, b5 = mlp(0.1)
    hist6, b6 = mlp(0.5)

    plt.plot(b1.history['loss'])
    plt.plot(b2.history['loss'])
    plt.plot(b3.history['loss'])
    plt.plot(b4.history['loss'])
    plt.plot(b5.history['loss'])
    plt.plot(b6.history['loss'])

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("MLP loss for different learning rates")
    plt.legend(['LR = 0.001', 'LR = 0.005', 'LR = 0.01', 'LR = 0.05', 'LR = 0.1', 'LR = 0.5'], loc='upper left')
    plt.show()


    plt.plot(b1.history['acc'])
    plt.plot(b2.history['acc'])
    plt.plot(b3.history['acc'])
    plt.plot(b4.history['acc'])
    plt.plot(b5.history['acc'])
    plt.plot(b6.history['acc'])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("MLP accuracy for different learning rates")
    plt.legend(['LR = 0.001', 'LR = 0.005', 'LR = 0.01', 'LR = 0.05', 'LR = 0.1', 'LR = 0.5'], loc='upper left')
    plt.show()



def autoencoders():
    numOfDigits = 5000
    input_img = Input(shape=(28, 28, 1))
    x_enc_dex = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x_enc_dex = MaxPooling2D((2, 2), padding='same')(x_enc_dex)
    x_enc_dex = Conv2D(8, (3, 3), activation='relu', padding='same')(x_enc_dex)
    x_enc_dex = MaxPooling2D((2, 2), padding='same')(x_enc_dex)
    x_enc_dex = Conv2D(8, (3, 3), activation='relu', padding='same')(x_enc_dex)
    x_enc_dex = MaxPooling2D((2, 2), padding='same')(x_enc_dex)
    x_enc_dex = MaxPooling2D((2, 2), padding='same')(x_enc_dex)
    x_enc_dex = Conv2D(1, (1, 1), activation='relu', padding='same')(x_enc_dex)
    encoded = MaxPooling2D((2, 1), padding='same', name='finish')(x_enc_dex)

    x_enc_dex = UpSampling2D((2, 1))(encoded)
    x_enc_dex = Conv2D(8, (1, 1), activation='relu', padding='same')(x_enc_dex)
    x_enc_dex = UpSampling2D((2, 2))(x_enc_dex)
    x_enc_dex = Conv2D(8, (3, 3), activation='relu', padding='same')(x_enc_dex)
    x_enc_dex = UpSampling2D((2, 2))(x_enc_dex)
    x_enc_dex = Conv2D(8, (3, 3), activation='relu', padding='same')(x_enc_dex)
    x_enc_dex = UpSampling2D((2, 2))(x_enc_dex)
    x_enc_dex = Conv2D(16, (3, 3), activation='relu')(x_enc_dex)
    x_enc_dex = UpSampling2D((2, 2))(x_enc_dex)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x_enc_dex)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    i = np.random.randint(0, 60000, numOfDigits)
    x_train = x_train[i]
    y_train = y_train[i]
    autoencoder.fit(x_train, x_train,
            shuffle=True,
            epochs=30,
            batch_size=128,
            validation_data=(x_test, x_test))
    intermediate_layer = keras.models.Model(inputs=autoencoder.input,
                                               outputs=autoencoder.get_layer('finish').output)

    points = intermediate_layer.predict(x_train)

    x_scatter = np.reshape(points[:,:,0,:], (numOfDigits,1))
    y_scatter = np.reshape(points[:,:,1,:], (numOfDigits,1))
    data_sample_labels = np.reshape(y_train, (numOfDigits,1))

    plt.figure(figsize=(6, 6))
    plt.scatter(x_scatter, y_scatter, c=data_sample_labels)
    plt.colorbar()
    plt.show()

