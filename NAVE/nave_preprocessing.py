# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:31:12 2022

@author: Helard.Becerra
"""

import skvideo
import skvideo.io
import skvideo.datasets
import skvideo.utils
import skvideo.measure

import numpy as np
import pandas as pd
import os

def extract_brisque_features_nd(name_video,nframes):
    #skvideo.setFFmpegPath('C:\\FFMPEG\\ffmpeg.exe')
    vid = skvideo.io.vread(name_video, num_frames=nframes)
    features = skvideo.measure.brisque_features(vid[:,:,:,1])
    return features

def extract_brisque_features(name_video,h,w,nframes):
    #skvideo.setFFmpegPath('C:\\FFMPEG\\ffmpeg.exe')
    vid = skvideo.io.vread(name_video,h,w, num_frames=nframes)
    features = skvideo.measure.brisque_features(vid[:,:,:,1])
    return features

def frame_pooling(feature_matrix):
    # check number of features
    mean_vec = feature_matrix.mean(axis=0)
    #std_vec = feature_matrix.std(axis=0)
    #feature_vec = np.concatenate((mean_vec,std_vec),axis=0)
    return mean_vec#feature_vec

#def frame_pooling_custom(feature_matrix, frames):
#    # check number of features
#    mean_mat = feature_matrix[::frames]
#    #std_vec = feature_matrix.std(axis=0)
#    #feature_vec = np.concatenate((mean_vec,std_vec),axis=0)
#    return mean_mat

def build_train_dataset(csv_file, mediapath):
    # Empty dataset array
    datasetList = list()
    # read csv file
    videoDF = pd.read_csv(csv_file, sep=';')
    for ind in videoDF.index:
        # get video file name
        file = videoDF['testFile'][ind] + '.avi'
        file_name = os.path.join(mediapath, file)
        # Extract features and compute feature vector
        features = extract_brisque_features(file_name,0)
        vecfeatures = frame_pooling(features)
        # Concatenate vectors
        datasetList.append(vecfeatures)
    # Build DF with features and mos
    feature_matrix = np.vstack(datasetList)
    datasetDF = pd.DataFrame(data=feature_matrix)
    datasetDF['mos'] = videoDF['mos']
    # Create quality groups for distribution
    cut_points = [0.9999, np.quantile(datasetDF['mos'].values, .20), np.quantile(datasetDF['mos'].values, .40), np.quantile(datasetDF['mos'].values, .60), np.quantile(datasetDF['mos'].values, .80), 5.000001] 
    categ = pd.cut(datasetDF.mos, bins=cut_points, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    datasetDF['mos_group'] = categ
    
    return datasetDF

def build_train_dataset_frame(csv_file, mediapath):
    # Empty dataset array
    num_frames = 300
    datasetList = list()
    refList = list()
    testList = list()
    frameList = list()
    # DF columns names
    feature_labels = ['f'+str(i) for i in range(36)]
    # read csv file
    videoDF = pd.read_csv(csv_file, sep=';')
    for ind in videoDF.index:
        # get video file name
        file = videoDF['testFile'][ind] + '.avi'
        file_name = os.path.join(mediapath, file)
        # Extract features and compute feature vector
        features = extract_brisque_features(file_name,num_frames)
        print('Features extracted ' + str(ind))
        # Add vectors
        refFile_vector = np.full((num_frames, 1), videoDF['refFile'][ind])
        testFile_vector = np.full((num_frames, 1), videoDF['testFile'][ind])
        mos_vector = np.full((num_frames, 1), videoDF['mos'][ind])
        
        features = np.append(features,mos_vector, axis=1)
        
        # Concatenate vectors
        datasetList.append(features)
        refList.append(refFile_vector)
        testList.append(testFile_vector)
        frameList.append(np.arange(num_frames))
    # Build DF with features and mos
    feature_matrix = np.vstack(datasetList)
    datasetDF = pd.DataFrame(data=feature_matrix, columns=feature_labels + ['mos'])
    # Create quality groups for distribution
    cut_points = [0.9999, np.quantile(datasetDF['mos'].values, .20), np.quantile(datasetDF['mos'].values, .40), np.quantile(datasetDF['mos'].values, .60), np.quantile(datasetDF['mos'].values, .80), 5.000001] 
    categ = pd.cut(datasetDF.mos, bins=cut_points, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    datasetDF['mos_group'] = categ
    refFile_column = np.vstack(refList)
    testFile_column = np.vstack(testList)
    frame_column = np.hstack(frameList)
    datasetDF['refFile'] = refFile_column
    datasetDF['testFile'] = testFile_column
    datasetDF['nframe'] = frame_column
    
    return datasetDF
    

