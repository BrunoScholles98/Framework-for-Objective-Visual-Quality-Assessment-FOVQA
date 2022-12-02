# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:53:18 2022

@author: Helard.Becerra
"""

from AE_functions import *
from nave_preprocessing import *

from tensorflow import keras

# input variables
models_path = 'saved_models/'
test_file = 'Seeking_30_480_1050.mp4'

# load models
mapping_model = keras.models.load_model(models_path + 'mapping.h5')
enc1_model = keras.models.load_model(models_path + 'enc1.h5')
enc2_model = keras.models.load_model(models_path + 'enc2.h5')

# mos prediction
predicted_mos = float(nave_predict(test_file, mapping_model, enc1_model, enc2_model)[0])