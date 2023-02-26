import skvideo.measure
import skvideo.io
import numpy as np
import subprocess as sp
import metrikz
import cv2
import matplotlib.image as mpimg
import os
from utils import tools
from NAVE import nave_preprocessing
from NAVE import AE_functions
from tensorflow import keras

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------Skvideo Full Reference--------------------------------------------
#-----------------------------------------------------------------------------------------------------------
def measureSkvideoMetric(metric_func, metric_name, videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    reference = f"{refpath}{videoRef}{'.'}{formato}"
    dist = f"{distpath}{videoDist}{'.'}{formato}"

    print(f"Metric: {metric_name}")
    print(f"Reference: {videoRef}")
    print(f"Distortion: {videoDist}")

    ref = skvideo.io.vread(reference, as_grey=True)
    dis = skvideo.io.vread(dist, as_grey=True)

    scores = metric_func(ref, dis)
    avg_score = np.mean(scores)

    print(f"AvScore: {avg_score}")
    print('\n\n')
    return avg_score

def measureSSIM(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    return measureSkvideoMetric(skvideo.measure.ssim, 'SSIM', videoRef, videoDist, formato, refpath, distpath, vmaf_model)

def measureMSSSIM(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    return measureSkvideoMetric(skvideo.measure.msssim, 'MSSSIM', videoRef, videoDist, formato, refpath, distpath, vmaf_model)

def measureMSE(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    return measureSkvideoMetric(skvideo.measure.mse, 'MSE', videoRef, videoDist, formato, refpath, distpath, vmaf_model)

def measurePSNR(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    return measureSkvideoMetric(skvideo.measure.psnr, 'PSNR', videoRef, videoDist, formato, refpath, distpath, vmaf_model)

#---------------------------------------------------------------------------------------------------------
#-----------------------------------------Skvideo No Reference--------------------------------------------
#---------------------------------------------------------------------------------------------------------
def measureNIQE(videoDist, formato, distpath, vmaf_model):   
    print('Metric: NIQE')
    print(f"{'Distortion: '}{videoDist}")

    dist = f"{distpath}{videoDist}{'.'}{formato}"
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.niqe(dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#-----------------------------------------------------------------------------------------
#-----------------------------------------VMAF--------------------------------------------
#-----------------------------------------------------------------------------------------
def measureVMAF(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    command1 = f"{'ffmpeg -i '}{distpath}"
    distVideo = videoDist
    command2 = f"{' -i '}{refpath}"
    refVideo = videoRef
    command3 = ' -lavfi libvmaf="model_path=' + vmaf_model + '" -f null -'
    command = f"{command1}{distVideo}{'.'}{formato}{command2}{refVideo}{'.'}{formato}{command3}"

    print('Metric: VMAF')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    output = sp.getoutput(command)
    print(output)
    splitter = 'VMAF score: '
    score = output.split(splitter)[1]

    print(f"{'VMAF Score: '}{score}")
    print('\n\n')
    return score

#-------------------------------------------------------------------------------------------------
#--------------------------------------------PyMetrikz--------------------------------------------
#-------------------------------------------------------------------------------------------------
def measurePyMetrikz(videoRef, videoDist, formato, refpath, distpath, vmaf_model, metric_func):

    print(f"{'Metric: '}{metric_func.__name__.upper()}")
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    metricResults = []
    path = './frames'

    stringPathRef = f"{refpath}{videoRef}{'.'}{formato}"
    stringPathDist = f"{distpath}{videoDist}{'.'}{formato}"

    ref = cv2.VideoCapture(stringPathRef)
    dist = cv2.VideoCapture(stringPathDist)

    successRef,framesRef = ref.read()
    successDist,framesDist = dist.read()

    countRef = 0
    countDist = 0
    i = 0

    while successRef & successDist: 
        cv2.imwrite(os.path.join(path ,'frameRef%d.png') % countRef, framesRef)    
        successRef,framesRef = ref.read()
        cv2.imwrite(os.path.join(path ,'frameDist%d.png') % countDist, framesDist)      
        successDist,framesDist = dist.read()
        pathFrameRef = f"{'./frames/frameRef'}{i}{'.png'}"
        pathDistRef = f"{'./frames/frameDist'}{i}{'.png'}"
        refFrameMeasure = mpimg.imread(pathFrameRef)
        distFrameMeasure = mpimg.imread(pathDistRef)
        value = metric_func(refFrameMeasure, distFrameMeasure) 
        metricResults.append(value)    
        countRef += 1
        countDist += 1
        i += 1

    tools.cleanFrameFolder()

    score = np.mean(metricResults)
    print(f"{metric_func.__name__.upper()} Score: {score}")
    print('\n\n')
    
    return score

def measureRMSE(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, formato, refpath, distpath, vmaf_model, metrikz.rmse)

def measureSNR(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, formato, refpath, distpath, vmaf_model, metrikz.snr)

def measureWSNR(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, formato, refpath, distpath, vmaf_model, metrikz.wsnr)

def measureUQI(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, formato, refpath, distpath, vmaf_model, metrikz.uqi)

def measurePBVIF(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, formato, refpath, distpath, vmaf_model, metrikz.pbvif)

#-----------------------------------------------------------------------------------------
#-----------------------------------------NAVE--------------------------------------------
#-----------------------------------------------------------------------------------------
def measureNAVE(videoDist, formato, distpath, vmaf_model):
    print('Metric: NAVE')
    print(f"{'Distortion: '}{videoDist}")

    models_path = os.getcwd() + '/NAVE/saved_models/'
    test_file = f"{distpath}{videoDist}{formato}"

    # load models
    mapping_model = keras.models.load_model(models_path + 'mapping.h5')
    enc1_model = keras.models.load_model(models_path + 'enc1.h5')
    enc2_model = keras.models.load_model(models_path + 'enc2.h5')

    # mos prediction
    predicted_mos = float(AE_functions.nave_predict_nd(test_file, mapping_model, enc1_model, enc2_model)[0])

    print(f"{'AvScore: '}{predicted_mos}")
    print('\n\n')
    return predicted_mos