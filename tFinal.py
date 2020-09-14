import numpy as np
import cv2
import glob
from image2feature import Extractor
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt




def entrenamiento_redes_neuronales():
    x_test, y_test = list(), list()
    x_train, y_train = list(), list()

    dataset = np.loadtxt('dataset.txt', dtype='str', delimiter=' ', usecols=range(3))
    #dataset = dataset[1:]
    for sujeto in dataset:
        cant_img = len(glob.glob('Cohn-kanade-dataset/' + sujeto[0] + '/' + sujeto[1] + '/*.png'))
        mitad = cant_img / 2
        mitad=int(mitad)
        # print mitad

        preproc=Extractor()

        for i in range(mitad, cant_img, 1):
            if (i < 10):
                ruta = sujeto[0] + '/' + sujeto[1] + '/' + sujeto[0] + '_' + sujeto[1] + '_0000000' + str(i) + '.png'
            else:
                ruta = sujeto[0] + "/" + sujeto[1] + "/" + sujeto[0] + "_" + sujeto[1] + '_000000' + str(i) + '.png'

            img = cv2.imread('Cohn-kanade-dataset/' + ruta)

            if(i==cant_img-2):
                caracteristicas=preproc.extract(img)
                #print(len(caracteristicas))
                x_test.append(caracteristicas)
                y_test.append(sujeto[2])

            else:
                caracteristicas=preproc.extract(img)
                #print(len(caracteristicas))

                x_train.append(caracteristicas)
                y_train.append(sujeto[2])

    x_train, y_train = np.array(x_train), np_utils.to_categorical(y_train)
    x_test, y_test = np.array(x_test), np_utils.to_categorical(y_test)

    with gzip.open('train2.pkl', 'wb') as f:
        pickle.dump((x_train,y_train), f, pickle.HIGHEST_PROTOCOL)
    with gzip.open('test2.pkl', 'wb') as f:
        pickle.dump((x_test, y_test), f, pickle.HIGHEST_PROTOCOL)





def ModeloSecuencial():

    f=gzip.open('train2.pkl','rb')
    x_train,y_train= pickle.load(f)

    """
    print(x_train)
    print("------------------------------")
    print(y_train)
    print("------------------------------")
    print("tam_x_train="+str(np.shape(x_train)))
    print("tam_y_train=" + str(np.shape(y_train)))
    """

    g=gzip.open('test2.pkl','rb')
    x_test,y_test= pickle.load(g)

    """
    print("------------------------------")
    print(x_test)
    print("------------------------------")
    print(y_test)
    print("------------------------------")
    print("tam_x_test=" + str(np.shape(x_test)))
    print("tam_y_test=" + str(np.shape(y_test)))
    """



    n = 2690   # Lo normalizamos
    x_train = x_train[:, 0:n] / np.amax(x_train)
    x_test = x_test[:, 0:n] / np.amax(x_test)


    model = Sequential()

    model.add(Dense(1470, input_dim=n, activation='relu'))
    model.add(Dense(735, activation='relu'))
    model.add(Dense(368, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(92, activation='relu'))
    model.add(Dense(7, activation='sigmoid'))


    epocas = 100
    #lrate = 0.01
    #decay = lrate / epocas
    #sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epocas, batch_size=20)

    score = model.evaluate(x_test, y_test)

    print('\n>>>Esto es presicion: \n %s: %.2f%%' % (model.metrics_names[1], score[1] * 100))
    max_class_test = np.argmax(y_test, axis=1)  # Nuestro Gran True







#entrenamiento_redes_neuronales()
ModeloSecuencial()

#optimizador  optimizer="adam"
