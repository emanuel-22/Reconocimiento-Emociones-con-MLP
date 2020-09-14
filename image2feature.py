# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:11:35 2019

@author: Ismael Orozco

"""

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import numpy as np
import cv2

class Extractor():
    def __init__(self):
        self.modelo = VGG16(weights='imagenet', include_top=False)

    def extract(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LINEAR)
        image = np.array(image).reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        features = self.modelo.predict(image)
        features = features.flatten()
        return features
        
'''
Como utilizar:

from image2feature import Extractor
preproc = Extractor()
... 

    feature = preproc.extract(image)
... 
'''