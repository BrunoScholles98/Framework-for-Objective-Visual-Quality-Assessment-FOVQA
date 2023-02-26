import skvideo.measure
import skvideo.io
import numpy as np
import subprocess as sp
import metrikz
import cv2
import matplotlib.image as mpimg
import os
from utils import tools
from NAVE import AE_functions
from NAVE import nave_preprocessing
from tensorflow import keras

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------Skvideo Full Reference--------------------------------------------
#-----------------------------------------------------------------------------------------------------------
def measureSkvideoMetric(metric_func, metric_name, videoRef, videoDist, h, w, refpath, distpath):
    # Create strings containing paths to reference and distorted files
    reference = f"{refpath}{videoRef}{'.yuv'}"
    dist = f"{distpath}{videoDist}{'.yuv'}"

    # Print information about the metric and videos being compared
    print(f"Metric: {metric_name}")
    print(f"Reference: {videoRef}")
    print(f"Distortion: {videoDist}")

    # Load reference and distorted videos into memory as numpy arrays using scikit-video library
    ref = skvideo.io.vread(reference, h, w, as_grey=True)
    dis = skvideo.io.vread(dist, h, w, as_grey=True)

    # Calculate video quality metric score using the function provided as argument and the average score
    scores = metric_func(ref, dis)
    avg_score = np.mean(scores)

    print(f"AvScore: {avg_score}")
    print('\n\n')

    return avg_score

def measureSSIM(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    return measureSkvideoMetric(skvideo.measure.ssim, 'SSIM', videoRef, videoDist, h, w, refpath, distpath)

def measureMSSSIM(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    return measureSkvideoMetric(skvideo.measure.msssim, 'MSSSIM', videoRef, videoDist, h, w, refpath, distpath)

def measureMSE(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    return measureSkvideoMetric(skvideo.measure.mse, 'MSE', videoRef, videoDist, h, w, refpath, distpath)

def measurePSNR(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    return measureSkvideoMetric(skvideo.measure.psnr, 'PSNR', videoRef, videoDist, h, w, refpath, distpath)

#---------------------------------------------------------------------------------------------------------
#-----------------------------------------Skvideo No Reference--------------------------------------------
#---------------------------------------------------------------------------------------------------------
def measureNIQE(videoDist, h, w, distpath, vmaf_model):   
    print('Metric: NIQE')
    print(f"{'Distortion: '}{videoDist}")

    dist = f"{distpath}{videoDist}{'.yuv'}"
    dis = skvideo.io.vread(dist, h, w, as_grey=True)    
    
    scores = skvideo.measure.niqe(dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#-----------------------------------------------------------------------------------------
#-----------------------------------------VMAF--------------------------------------------
#-----------------------------------------------------------------------------------------
def measureVMAF(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    # Construct FFmpeg command to calculate VMAF score for given videos
    command = f"ffmpeg -video_size {w}x{h} -i {distpath}{videoDist}.yuv -video_size {w}x{h} -i {refpath}{videoRef}.yuv -lavfi libvmaf=model_path={vmaf_model} -f null -"

    # Print information about the metric and videos being compared
    print('Metric: VMAF')
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")

    # Execute FFmpeg command and capture the output
    output = sp.getoutput(command)

    # Split the output to extract the VMAF score
    splitter = 'VMAF score: '
    score = output.split(splitter)[1]

    print(f"{'VMAF Score: '}{score}")
    print('\n\n')

    return score

#-------------------------------------------------------------------------------------------------
#--------------------------------------------PyMetrikz--------------------------------------------
#-------------------------------------------------------------------------------------------------
def measurePyMetrikz(videoRef, videoDist, h, w, refpath, distpath, vmaf_model, metric_func):
    # Print metric name and video names
    print(f"{'Metric: '}{metric_func.__name__.upper()}")
    print(f"{'Reference: '}{videoRef}")
    print(f"{'Distortion: '}{videoDist}")
    
    # Create an empty list to hold the metric results for each frame
    metricResults = []

    # Set the path for the folder to store the frames
    path = './frames'

    # Convert the reference and distorted videos to AVI format
    tools.convertionToAVI(videoRef, h, w, refpath)
    tools.convertionToAVI(videoDist, h, w, distpath)

    # Set the path for the reference and distorted videos in AVI format
    stringPathRef = f"{'./videosAVI/'}{videoRef}{'.avi'}"
    stringPathDist = f"{'./videosAVI/'}{videoDist}{'.avi'}"

    # Create OpenCV video capture objects for the reference and distorted videos
    ref = cv2.VideoCapture(stringPathRef)
    dist = cv2.VideoCapture(stringPathDist)

    # Read the first frames from the reference and distorted videos
    successRef,framesRef = ref.read()
    successDist,framesDist = dist.read()

    # Set counters for the frames in the reference and distorted videos, and an index counter
    countRef = 0
    countDist = 0
    i = 0

    # Loop through the frames in the reference and distorted videos
    while successRef & successDist: 

        # Write the current frames to the folder for the frames
        cv2.imwrite(os.path.join(path ,'frameRef%d.png') % countRef, framesRef)    
        successRef,framesRef = ref.read()
        cv2.imwrite(os.path.join(path ,'frameDist%d.png') % countDist, framesDist)      
        successDist,framesDist = dist.read()

        # Set the paths for the current frames in the folder for the frames
        pathFrameRef = f"{'./frames/frameRef'}{i}{'.png'}"
        pathDistRef = f"{'./frames/frameDist'}{i}{'.png'}"

        # Read the current frames from the folder for the frames using matplotlib image
        refFrameMeasure = mpimg.imread(pathFrameRef)
        distFrameMeasure = mpimg.imread(pathDistRef)

        # Compute the metric value for the current frames using the specified metric function
        value = metric_func(refFrameMeasure, distFrameMeasure) 

        # Append the metric value to the list of metric results
        metricResults.append(value)    

        # Increment the counters for the frames in the reference and distorted videos, and the index counter
        countRef += 1
        countDist += 1
        i += 1

    # Clean up the folder for the frames
    tools.cleanFrameFolder()

    # Compute the average metric value across all frames
    score = np.mean(metricResults)
    
    print(f"{metric_func.__name__.upper()} Score: {score}")
    print('\n\n')    

    return score

def measureRMSE(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, h, w, refpath, distpath, vmaf_model, metrikz.rmse)

def measureSNR(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, h, w, refpath, distpath, vmaf_model, metrikz.snr)

def measureWSNR(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, h, w, refpath, distpath, vmaf_model, metrikz.wsnr)

def measureUQI(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, h, w, refpath, distpath, vmaf_model, metrikz.uqi)

def measurePBVIF(videoRef, videoDist, h, w, refpath, distpath, vmaf_model):
    return measurePyMetrikz(videoRef, videoDist, h, w, refpath, distpath, vmaf_model, metrikz.pbvif)

#-----------------------------------------------------------------------------------------
#-----------------------------------------NAVE--------------------------------------------
#-----------------------------------------------------------------------------------------
def measureNAVE(videoDist, h, w, distpath, vmaf_model):
    print('Metric: NAVE')
    print(f"{'Distortion: '}{videoDist}")

    models_path = os.getcwd() + '/NAVE/saved_models/'
    test_file = f"{distpath}{videoDist}{'.yuv'}"

    # load models
    mapping_model = keras.models.load_model(models_path + 'mapping.h5')
    enc1_model = keras.models.load_model(models_path + 'enc1.h5')
    enc2_model = keras.models.load_model(models_path + 'enc2.h5')

    # mos prediction
    predicted_mos = float(AE_functions.nave_predict(test_file, h,w,mapping_model, enc1_model, enc2_model)[0])

    print(f"{'AvScore: '}{predicted_mos}")
    print('\n\n')
    return predicted_mos