# import libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset():
    # load cifar10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_train, x_test, y_train, y_test

def view_data(trainx, trainy, testx, testy):
    # print dataset shape
    print('Train: X=%s, Y=%s' % (trainx.shape, trainy.shape))
    print('Test: X=%s, Y=%s' % (testx.shape, testy.shape))

    '''
    # plot data to view first 9 images
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(trainx[i])
    plt.show()
    '''

def prep_pixels(trainx, testx, trainy, testy):
    # normalize the data by first converting to float then dividing by 255
    train = trainx.astype('float32')
    test = testx.astype('float32')
    train_norm_x = train / 255
    test_norm_x = test / 255
    # convert test data to one hot encoded
    y_test = to_categorical(testy)
    y_train = to_categorical(trainy)
    return train_norm_x, test_norm_x, y_train, y_test

def define_model(x_train, y_train, x_test, y_test):
    # define your model
    model = tf.keras.Sequential()
    # first convolutional layer
    model.add(Conv2D(32,(3,3), input_shape= x_train.shape[1:], padding='same'))
    model.add(Activation('relu'))
    # we use batch normalizatin so that input that goes through is normalized
    model.add(BatchNormalization())
    # we add dropout as 0.2 so it will remove 20% of existing connections, this prevents overfitting
    model.add(Conv2D(32,(3,3), input_shape= x_train.shape[1:], padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # second conv layer with increased size so that model can learn more intricate features
    model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    # we add one pooling layer here
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    # we can add as many conv layers as we want but have to make sure we increase their filter size
    # don't use too much pooling layer as it might deviate the model to make any sense of image
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    # now we'll flatten or model i.e convert it into vector
    model.add(Flatten())
    # we'll make use of dense import and create first fully connected layer
    model.add(Dense(128, kernel_constraint=max_norm(3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_constraint=max_norm(3), activation='relu'))
    model.add(BatchNormalization())
    # in final layer we pass number of classes as it will output those values for that classes and add a softmax activation
    model.add(Dense(10))
    model.add(Activation('softmax'))
    # now we will compile the model
    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    print(model.summary())
    return model

def summary_plot(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='test')
    # plot Accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='red', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


def train_model():
    seed = 23
    # loading the dataset
    x_train, x_test, y_train, y_test = load_dataset()
    # displaying the data
    view_data(x_train, y_train, x_test, y_test)
    # pre processing of data
    x_train, x_test, y_train, y_test = prep_pixels(x_train, x_test, y_train, y_test)
    # creating model
    model = define_model(x_train, y_train, x_test, y_test)


    '''''
    # Data generator basically generates copy of data hence increases the training data with slight random modifications
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    # prepare iterator
    it_train = datagen.flow(x_train, y_train, batch_size=64)
    steps = int(x_train.shape[0] / 64)
    '''''

    # fit model
    history = model.fit(x_train, y_train, epochs=400, validation_data=(x_test, y_test), verbose=2)
    # evaluate model
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('> %.3f' % (acc*100.0))
    # plotting summary of model and saving in file
    summary_plot(history)
    # saving this model
    model.save('object_recognition_model.h5')

train_model()








