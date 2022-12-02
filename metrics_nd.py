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

def measureSSIM(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    reference = f"{refpath}{videoRef}{'.'}{formato}"
    dist = f"{distpath}{videoDist}{'.'}{formato}"

    print('Metric: SSIM')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    ref = skvideo.io.vread(reference, as_grey=True)
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.ssim(ref, dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#------------------------------------------------------------------------------------------------

def measureMSSSIM(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    reference = f"{refpath}{videoRef}{'.'}{formato}"
    dist = f"{distpath}{videoDist}{'.'}{formato}"

    print('Metric: MSSSIM')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    ref = skvideo.io.vread(reference, as_grey=True)
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.msssim(ref, dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#------------------------------------------------------------------------------------------------

def measureMSE(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    reference = f"{refpath}{videoRef}{'.'}{formato}"
    dist = f"{distpath}{videoDist}{'.'}{formato}"

    print('Metric: MSE')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    ref = skvideo.io.vread(reference, as_grey=True)
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.mse(ref, dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#------------------------------------------------------------------------------------------------

def measurePSNR(videoRef, videoDist, formato, refpath, distpath, vmaf_model):
    reference = f"{refpath}{videoRef}{'.'}{formato}"
    dist = f"{distpath}{videoDist}{'.'}{formato}"

    print('Metric: PSNR')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    ref = skvideo.io.vread(reference, as_grey=True)
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.psnr(ref, dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------

def measureRMSE(videoRef, videoDist, formato, refpath, distpath, vmaf_model):

    print('Metric: RMSE')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    metricResults = []
    path = './frames'                                                            

    nomeRef = videoRef
    nomeDist = videoDist

    stringPathRef = f"{refpath}{nomeRef}{'.'}{formato}"
    stringPathDist = f"{distpath}{nomeDist}{'.'}{formato}"

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
        value = metrikz.rmse(refFrameMeasure,distFrameMeasure) 
        metricResults.append(value)    
        countRef += 1
        countDist += 1
        i += 1

    tools.cleanFrameFolder()

    score = np.mean(metricResults)
    print(f"{'RMSE Score: '}{score}")
    print('\n\n')
    
    return score

#------------------------------------------------------------------------------------------------

def measureSNR(videoRef, videoDist, formato, refpath, distpath, vmaf_model):

    print('Metric: SNR')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    metricResults = []
    path = './frames'                                                            

    nomeRef = videoRef
    nomeDist = videoDist

    stringPathRef = f"{refpath}{nomeRef}{'.'}{formato}"
    stringPathDist = f"{distpath}{nomeDist}{'.'}{formato}"

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
        value = metrikz.snr(refFrameMeasure,distFrameMeasure) 
        metricResults.append(value)    
        countRef += 1
        countDist += 1
        i += 1

    tools.cleanFrameFolder()

    score = np.mean(metricResults)
    print(f"{'SNR Score: '}{score}")
    print('\n\n')
    
    return score
    
#------------------------------------------------------------------------------------------------

def measureWSNR(videoRef, videoDist, formato, refpath, distpath, vmaf_model):

    print('Metric: WSNR')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    metricResults = []
    path = './frames'                                                            

    nomeRef = videoRef
    nomeDist = videoDist

    stringPathRef = f"{refpath}{nomeRef}{'.'}{formato}"
    stringPathDist = f"{distpath}{nomeDist}{'.'}{formato}"

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
        value = metrikz.wsnr(refFrameMeasure,distFrameMeasure) 
        metricResults.append(value)    
        countRef += 1
        countDist += 1
        i += 1

    tools.cleanFrameFolder()

    score = np.mean(metricResults)
    print(f"{'WSNR Score: '}{score}")
    print('\n\n')
    
    return score

#------------------------------------------------------------------------------------------------

def measureUQI(videoRef, videoDist, formato, refpath, distpath, vmaf_model):

    print('Metric: UQI')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    metricResults = []
    path = './frames'                                                            

    nomeRef = videoRef
    nomeDist = videoDist


    stringPathRef = f"{refpath}{nomeRef}{'.'}{formato}"
    stringPathDist = f"{distpath}{nomeDist}{'.'}{formato}"

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
        value = metrikz.uqi(refFrameMeasure,distFrameMeasure) 
        metricResults.append(value)    
        countRef += 1
        countDist += 1
        i += 1

    tools.cleanFrameFolder()

    score = np.mean(metricResults)
    print(f"{'UQI Score: '}{score}")
    print('\n\n')
    
    return score
    
#------------------------------------------------------------------------------------------------

def measurePBVIF(videoRef, videoDist, formato, refpath, distpath, vmaf_model):

    print('Metric: PBVIF')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    metricResults = []
    path = './frames'                                                            

    nomeRef = videoRef
    nomeDist = videoDist

    stringPathRef = f"{refpath}{nomeRef}{'.'}{formato}"
    stringPathDist = f"{distpath}{nomeDist}{'.'}{formato}"

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
        value = metrikz.pbvif(refFrameMeasure,distFrameMeasure) 
        metricResults.append(value)    
        countRef += 1
        countDist += 1
        i += 1

    tools.cleanFrameFolder()

    score = np.mean(metricResults)
    print(f"{'PBVIF Score: '}{score}")
    print('\n\n')
    
    return score

def measureNAVE(videoDist, formato, distpath, vmaf_model):
    print('Metric: NAVE')
    print(f"{'Distortion: '}{videoDist}")

    models_path = '/NAVE/saved_models/'
    test_file = f"{distpath}{videoDist}{formato}"

    # load models
    mapping_model = keras.models.load_model(models_path + 'mapping.h5')
    enc1_model = keras.models.load_model(models_path + 'enc1.h5')
    enc2_model = keras.models.load_model(models_path + 'enc2.h5')

    # mos prediction
    predicted_mos = float(nave_predict_nd(test_file, mapping_model, enc1_model, enc2_model)[0])

    print(f"{'AvScore: '}{predicted_mos}")
    print('\n\n')
    return predicted_mos