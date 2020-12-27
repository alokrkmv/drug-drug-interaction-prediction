'''

Author : Alok Kumar
Date : 27/12/2020
This file contains the code for the Dense neural network that is used as the learning and prediction layer in our project.
This model can be trained by running main file. The fine tuned model is also saved as keras model and can be hot reloaded for performing predictions.

### Model Summary ###

Layer (type)                 Output Shape              Param #
=================================================================
dense_3 (Dense)              (None, 64)                90432
_________________________________________________________________
dense_4 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_5 (Dense)              (None, 2)                 66


'''
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers.core import Dropout, Activation


def DNN(input_size=1000):
    model = Sequential()
    model.add(Dense(64, input_dim=1412, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model