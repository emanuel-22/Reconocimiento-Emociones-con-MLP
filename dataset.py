# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 19:40:00 2019

@author: Ismael Orozco
"""

import numpy as np
from keras.utils import np_utils



X_test, y_test = list(), list()        
X_train, y_train = list(), list()
...
    X_train.append(np.array(video))
    y_train.append(label)
...
# la segunda la modifica....
X_train, y_train = np.array(X_train), np_utils.to_categorical(y_train)
X_test, y_test = np.array(X_test), np_utils.to_categorical(y_test)


'''
para entrenar el modelo
    model = Sequential()
    ...
    model.add(Dense(7, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    categorical_crossentropy ya que son 7 clases!
'''