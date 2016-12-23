import cv2
import os, glob
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt


def load_data(data_dir):
    """
    From: https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.v471kaepx
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []

    category = 0
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        #stop = 0
        for f in file_names:
            img = cv2.imread(f)
            imresize = cv2.resize(img, (200, 125))
            #plt.imshow(imresize)
            images.append(imresize)
            labels.append(category)
            #if stop > 30:
            #    break
            #stop += 1
        category += 1

    return images, labels


def cross_validate(Xs, ys):
    X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def load_test_data():
    return

def makePrediction():
    return None

def createCNNModel(num_classes):
    # Create the model
    model = Sequential()
    model.add(Convolution2D(125, 3, 3, input_shape=(125, 200, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(125, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model, epochs

if __name__ == '__main__':
    # load data
    train_data_dir = "train/"
    images, labels = load_data(train_data_dir)
    print("Data Loaded")

    # cross validate
    X_train, X_test, y_train, y_test = cross_validate(images, labels)
    print("Data cross validated")

    # normalize inputs from 0-255 and 0.0-1.0
    X_train = np.array(X_train).astype('float32')
    X_test = np.array(X_test).astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    print("Data normalized and hot encoded.")

    # create our CNN model
    model, epochs = createCNNModel(num_classes)
    print("CNN Model created.")

    # fit and run our model
    seed = 7
    np.random.seed(seed)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    print "done"
