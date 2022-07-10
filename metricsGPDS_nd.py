import skvideo.measure
import skvideo.io
import numpy as np
import subprocess as sp
import metrikz
import cv2
import matplotlib.image as mpimg
import os
from utils import toolsGPDS

def measureSSIM(videoRef, videoDist, formato, refpath, distpath):
    reference = f"{refpath}{videoRef}{'.'}{formato}"
    dist = f"{distpath}{videoDist}{'.'}{formato}"

    print('Metrica: SSIM')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    ref = skvideo.io.vread(reference, as_grey=True)
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.ssim(ref, dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#------------------------------------------------------------------------------------------------

def measureMSSSIM(videoRef, videoDist, formato, refpath, distpath):
    reference = f"{refpath}{videoRef}{'.'}{formato}"
    dist = f"{distpath}{videoDist}{'.'}{formato}"

    print('Metrica: MSSSIM')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    ref = skvideo.io.vread(reference, as_grey=True)
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.msssim(ref, dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#------------------------------------------------------------------------------------------------

def measureMSE(videoRef, videoDist, formato, refpath, distpath):
    reference = f"{refpath}{videoRef}{'.'}{formato}"
    dist = f"{distpath}{videoDist}{'.'}{formato}"

    print('Metrica: MSE')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    ref = skvideo.io.vread(reference, as_grey=True)
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.mse(ref, dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#------------------------------------------------------------------------------------------------

def measurePSNR(videoRef, videoDist, formato, refpath, distpath):
    reference = f"{refpath}{videoRef}{'.'}{formato}"
    dist = f"{distpath}{videoDist}{'.'}{formato}"

    print('Metrica: PSNR')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    ref = skvideo.io.vread(reference, as_grey=True)
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.psnr(ref, dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#------------------------------------------------------------------------------------------------

def measureNIQE(videoDist, formato, distpath):   
    print('Metrica: NIQE')
    print(f"{'Distorcao: '}{videoDist}")

    dist = f"{distpath}{videoDist}{'.'}{formato}"
    dis = skvideo.io.vread(dist, as_grey=True)    
    
    scores = skvideo.measure.niqe(dis)    
    avg_score = np.mean(scores)

    print(f"{'AvScore: '}{avg_score}")
    print('\n\n')
    return avg_score

#------------------------------------------------------------------------------------------------

def measureVMAF(videoRef, videoDist, formato, refpath, distpath):
    comando1 = f"{'ffmpeg -i '}{distpath}"
    distVideo = videoDist
    comando2 = f"{' -i '}{refpath}"
    refVideo = videoRef
    comando3 = ' -lavfi libvmaf="model_path=/home/brunoscholles/Framework/AudioVisualMeter/models/vmaf_v0.6.1.json" -f null -'
    comando = f"{comando1}{distVideo}{'.'}{formato}{comando2}{refVideo}{'.'}{formato}{comando3}"

    print('Metrica: VMAF')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    output = sp.getoutput(comando)
    print(output)
    splitter = 'VMAF score: '
    score = output.split(splitter)[1]

    print(f"{'VMAF Score: '}{score}")
    print('\n\n')
    return score

#------------------------------------------------------------------------------------------------

def measureRMSE(videoRef, videoDist, formato, refpath, distpath):

    print('Metrica: RMSE')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    resultadoMetrica = []
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
        valor = metrikz.rmse(refFrameMeasure,distFrameMeasure) 
        resultadoMetrica.append(valor)    
        countRef += 1
        countDist += 1
        i += 1

    toolsGPDS.cleanFrameFolder()

    score = np.mean(resultadoMetrica)
    print(f"{'RMSE Score: '}{score}")
    print('\n\n')
    
    return score

#------------------------------------------------------------------------------------------------

def measureSNR(videoRef, videoDist, formato, refpath, distpath):

    print('Metrica: SNR')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    resultadoMetrica = []
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
        valor = metrikz.snr(refFrameMeasure,distFrameMeasure) #métrica a ser utilizada, aqui
        resultadoMetrica.append(valor)    
        countRef += 1
        countDist += 1
        i += 1

    toolsGPDS.cleanFrameFolder()

    score = np.mean(resultadoMetrica)
    print(f"{'SNR Score: '}{score}")
    print('\n\n')
    
    return score
    
#------------------------------------------------------------------------------------------------

def measureWSNR(videoRef, videoDist, formato, refpath, distpath):

    print('Metrica: WSNR')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    resultadoMetrica = []
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
        valor = metrikz.wsnr(refFrameMeasure,distFrameMeasure) #métrica a ser utilizada, aqui
        resultadoMetrica.append(valor)    
        countRef += 1
        countDist += 1
        i += 1

    toolsGPDS.cleanFrameFolder()

    score = np.mean(resultadoMetrica)
    print(f"{'WSNR Score: '}{score}")
    print('\n\n')
    
    return score

#------------------------------------------------------------------------------------------------

def measureUQI(videoRef, videoDist, formato, refpath, distpath):

    print('Metrica: UQI')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    resultadoMetrica = []
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
        valor = metrikz.uqi(refFrameMeasure,distFrameMeasure) #métrica a ser utilizada, aqui
        resultadoMetrica.append(valor)    
        countRef += 1
        countDist += 1
        i += 1

    toolsGPDS.cleanFrameFolder()

    score = np.mean(resultadoMetrica)
    print(f"{'UQI Score: '}{score}")
    print('\n\n')
    
    return score
    
#------------------------------------------------------------------------------------------------

def measurePBVIF(videoRef, videoDist, formato, refpath, distpath):

    print('Metrica: PBVIF')
    print(f"{'Referencia: '}{videoRef}")
    print(f"{'Distorcao: '}{videoDist}")
    
    resultadoMetrica = []
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
        valor = metrikz.pbvif(refFrameMeasure,distFrameMeasure) #métrica a ser utilizada, aqui
        resultadoMetrica.append(valor)    
        countRef += 1
        countDist += 1
        i += 1

    toolsGPDS.cleanFrameFolder()

    score = np.mean(resultadoMetrica)
    print(f"{'PBVIF Score: '}{score}")
    print('\n\n')
    
    return score