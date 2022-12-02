# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:21:51 2022

@author: Helard.Becerra
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

from nave_preprocessing import *

from keras.regularizers import l2, l1
#from keras.utils import np_utils

#import pandas as pd
#import numpy as np




def trainAutoencoder_classification(X_train, X_test, y_train, y_test):
    #Set parameters
    n_inputs = X_train.shape[1]
    #nodes = [50,20]
    
    # Training parameters
    num_epocs = 10
    sparsity_reg = None #l1(0.05)
    weight_reg = None #l2(0.001)
    batch_s = 32
    
    ## AutoEncoder 1 ##
    # define encoder input
    input_matrix = Input(shape=(n_inputs,))
      
    # AEncoder layer 1 
    # Visual: [90 - 50] - 20 
    # Audio: [21 - 18] - 10 
    e_1 = Dense(50, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(input_matrix)
    e_1 = BatchNormalization()(e_1)
    e_1 = LeakyReLU()(e_1)
    d_1 = Dense(n_inputs, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(e_1)
    d_1 = BatchNormalization()(d_1)
    d_1 = LeakyReLU()(d_1)
    aemodel_1 = Model(inputs=input_matrix, outputs=d_1)
    
    # Compile Autoencoder Model
    aemodel_1.compile(optimizer='adamax', loss='mse')
    
    # Fit the Autoencoder Model to reconstruct input
    history = aemodel_1.fit(X_train, X_train, epochs=num_epocs, batch_size=batch_s, verbose=2, validation_data=(X_test,X_test))
    
    # Extract the encoder model and discard the decoder
    e_model_1 = Model(inputs=input_matrix, outputs=e_1)
    
    # Encode Xs
    X_train_encode_1 = e_model_1.predict(X_train)
    X_test_encode_1 = e_model_1.predict(X_test)
    
    ## AutoEncoder 2 ##
    n_inputs = 50
    # define encoder input
    input_matrix = Input(shape=(n_inputs,))
    
    # AEncoder layer 2
    # Visual: 90 - [50 - 20]
    # Audio: 21 - [18 - 10]
    e_2 = Dense(20, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(input_matrix)
    e_2 = BatchNormalization()(e_2)
    e_2 = LeakyReLU()(e_2)
    d_2 = Dense(50, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(e_2)
    d_2 = BatchNormalization()(d_2)
    d_2 = LeakyReLU()(d_2)
    aemodel_2 = Model(inputs=input_matrix, outputs=d_2)
    
    # Compile Autoencoder Model
    aemodel_2.compile(optimizer='adamax', loss='mse')
    
    # Fit the Autoencoder Model to reconstruct input
    history = aemodel_2.fit(X_train_encode_1, X_train_encode_1, epochs=num_epocs, batch_size=batch_s, verbose=2, validation_data=(X_test_encode_1,X_test_encode_1))
    
    # Extract the encoder model and discard the decoder
    e_model_2 = Model(inputs=input_matrix, outputs=e_2)
    
    # Encode Xs
    X_train_encode_2 = e_model_2.predict(X_train_encode_1)
    X_test_encode_2 = e_model_2.predict(X_test_encode_1)
    
    ## Mapping Function ##
    # Visual: 90 - 50 - [20]
    # Audio: 21 - 18 - [10]
    n_inputs = 20
    # define mapping input
    Mapping_input = Input(shape=(n_inputs,))
    # output mapping layer
    Mapping_output = Dense(4, activation='softmax')(Mapping_input)
    
    # define model
    AE_model = Model(inputs=Mapping_input, outputs=Mapping_output)
    
    # compile classification model
    AE_model.compile(optimizer='adamax', loss='mse')
    AE_model.fit(X_train_encode_2, y_train, epochs=10, batch_size=batch_s)
    
    return AE_model, e_model_1, e_model_2

def trainAutoencoder_regression(X_train, X_test, y_train, y_test):
    #Set parameters
    n_inputs = X_train.shape[1]
    #nodes = [50,20]
    
    # Training parameters
    num_epocs = 10
    sparsity_reg = None #l1(0.05)
    weight_reg = None #l2(0.001)
    batch_s = 32
    
    ## AutoEncoder 1 ##
    # define encoder input
    input_matrix = Input(shape=(n_inputs,))
      
    # AEncoder layer 1 
    # Visual: [90 - 50] - 20 
    # Audio: [21 - 18] - 10 
    e_1 = Dense(50, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(input_matrix)
    e_1 = BatchNormalization()(e_1)
    e_1 = LeakyReLU()(e_1)
    d_1 = Dense(n_inputs, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(e_1)
    d_1 = BatchNormalization()(d_1)
    d_1 = LeakyReLU()(d_1)
    aemodel_1 = Model(inputs=input_matrix, outputs=d_1)
    
    # Compile Autoencoder Model
    aemodel_1.compile(optimizer='adamax', loss='mse')
    
    # Fit the Autoencoder Model to reconstruct input
    history = aemodel_1.fit(X_train, X_train, epochs=num_epocs, batch_size=batch_s, verbose=2, validation_data=(X_test,X_test))
    
    # Extract the encoder model and discard the decoder
    e_model_1 = Model(inputs=input_matrix, outputs=e_1)
    
    # Encode Xs
    X_train_encode_1 = e_model_1.predict(X_train)
    X_test_encode_1 = e_model_1.predict(X_test)
    
    ## AutoEncoder 2 ##
    n_inputs = 50
    # define encoder input
    input_matrix = Input(shape=(n_inputs,))
    
    # AEncoder layer 2
    # Visual: 90 - [50 - 20]
    # Audio: 21 - [18 - 10]
    e_2 = Dense(20, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(input_matrix)
    e_2 = BatchNormalization()(e_2)
    e_2 = LeakyReLU()(e_2)
    d_2 = Dense(50, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(e_2)
    d_2 = BatchNormalization()(d_2)
    d_2 = LeakyReLU()(d_2)
    aemodel_2 = Model(inputs=input_matrix, outputs=d_2)
    
    # Compile Autoencoder Model
    aemodel_2.compile(optimizer='adamax', loss='mse')
    
    # Fit the Autoencoder Model to reconstruct input
    history = aemodel_2.fit(X_train_encode_1, X_train_encode_1, epochs=num_epocs, batch_size=batch_s, verbose=2, validation_data=(X_test_encode_1,X_test_encode_1))
    
    # Extract the encoder model and discard the decoder
    e_model_2 = Model(inputs=input_matrix, outputs=e_2)
    
    # Encode Xs
    X_train_encode_2 = e_model_2.predict(X_train_encode_1)
    X_test_encode_2 = e_model_2.predict(X_test_encode_1)
    
    ## Mapping Function ##
    # Visual: 90 - 50 - [20]
    # Audio: 21 - 18 - [10]
    n_inputs = 20
    # define mapping input
    Mapping_input = Input(shape=(n_inputs,))
    # output mapping layer
    Mapping_output = Dense(1, activation='linear')(Mapping_input)
    
    # define model
    AE_model = Model(inputs=Mapping_input, outputs=Mapping_output)
    
    # compile classification model
    AE_model.compile(optimizer='adamax', loss='mse')
    AE_model.fit(X_train_encode_2, y_train, epochs=10, batch_size=batch_s)
    
    return AE_model, e_model_1, e_model_2


def trainAutoencoder_basic(X_train, X_test, nodes, **kwargs):
    #Set parameters
    n_inputs = X_train.shape[1]
    inv_nodes = nodes[::-1]
    y_train = kwargs['y_train']
    y_test = kwargs['y_test']
    
    # Training parameters
    num_epocs = 1500
    sparsity_reg = None #l1(0.05)
    weight_reg = None #l2(0.0001)
    
    ## Encoder ##
    # define encoder input
    input_matrix = Input(shape=(n_inputs,))
      
    # Encoder layers
    e = Dense(nodes[0], activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(input_matrix)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    
    for node in nodes[1:-1]:
        e = Dense(node, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
    
    # bottleneck
    n_bottleneck = nodes[-1]
    bottleneck = Dense(n_bottleneck, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(e)
    
    ## Decoder ##
    # Decoder, layer 1
    d = Dense(inv_nodes[1], activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    
    for node in inv_nodes[2:-1]:
        d = Dense(node, activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
    
    # Output layer
    output_matrix = Dense(n_inputs, activation='sigmoid', activity_regularizer=sparsity_reg, kernel_regularizer=weight_reg)(d)

    ## AutoEncoder ##
    # Define Autoencoder Model
    model = Model(inputs=input_matrix, outputs=output_matrix)
    
    # Compile Autoencoder Model
    model.compile(optimizer='adam', loss='mse')
    
    # Fit the Autoencoder Model to reconstruct input
    history = model.fit(X_train, X_train, epochs=num_epocs, batch_size=16, verbose=2, validation_data=(X_test,X_test))
    
    # Extract the encoder model and discard the decoder
    Encoder_model = Model(inputs=input_matrix, outputs=bottleneck)
    
    # Encode Xs
    X_train_encode = Encoder_model.predict(X_train)
    X_test_encode = Encoder_model.predict(X_test)
        
    # number of input columns
    n_inputs = X_train_encode.shape[1]
    # define input
    Mapping_Input = Input(shape=(n_inputs,))
    # output layer
    Mapping_output = Dense(1, activation='linear')(Mapping_Input)
        
    # define classification model
    AE_model = Model(inputs=Mapping_Input, outputs=Mapping_output)
        
    # compile classification model
    AE_model.compile(optimizer='adam', loss='mse')
    AE_model.fit(X_train_encode, y_train, epochs=num_epocs)
    
    return AE_model, Encoder_model

def nave_encoder_nd(video_name,model):
    # load video and extract features
    features = extract_brisque_features_nd(video_name,0)
    vec_feats = frame_pooling(features)
    # encode features
    encoded = model.predict(np.transpose(vec_feats.reshape((-1, 1))))
    return encoded

def nave_encoder(video_name,h,w,model):
    # load video and extract features
    features = extract_brisque_features(video_name,h,w,0)
    vec_feats = frame_pooling(features)
    # encode features
    encoded = model.predict(np.transpose(vec_feats.reshape((-1, 1))))
    return encoded

def nave_predict_nd(video_name, mapping_model, enc1_model, enc2_model):
    #models_path = 'saved_models/'
    # load models
    #mapping_model = keras.models.load_model(models_path + 'mapping.h5')
    #enc1_model = keras.models.load_model(models_path + 'enc1.h5')
    #enc2_model = keras.models.load_model(models_path + 'enc2.h5')

    # mos prediction
    encoded_feats = nave_encoder_nd(video_name,enc1_model)
    encoded_feats2 = enc2_model.predict(encoded_feats)
    predicted_mos = mapping_model.predict(encoded_feats2)
    return predicted_mos

def nave_predict(video_name, h,w, mapping_model, enc1_model, enc2_model):
    #models_path = 'saved_models/'
    # load models
    #mapping_model = keras.models.load_model(models_path + 'mapping.h5')
    #enc1_model = keras.models.load_model(models_path + 'enc1.h5')
    #enc2_model = keras.models.load_model(models_path + 'enc2.h5')

    # mos prediction
    encoded_feats = nave_encoder(video_name,h,w,enc1_model)
    encoded_feats2 = enc2_model.predict(encoded_feats)
    predicted_mos = mapping_model.predict(encoded_feats2)
    return predicted_mos

def nave_predict_raw_feats(feature_vector, mapping_model, enc1_model, enc2_model):
    #models_path = 'saved_models/'
    # load models
    #mapping_model = keras.models.load_model(models_path + 'mapping.h5')
    #enc1_model = keras.models.load_model(models_path + 'enc1.h5')
    #enc2_model = keras.models.load_model(models_path + 'enc2.h5')

    # mos prediction
    encoded_feats = enc1_model.predict(np.transpose(feature_vector.reshape((-1, 1))))
    encoded_feats2 = enc2_model.predict(encoded_feats)
    predicted_mos = mapping_model.predict(encoded_feats2)
    return predicted_mos