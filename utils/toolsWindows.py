import os, shutil
import subprocess as sp

def cleanFrameFolder():
    folder = './frames'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Falha ao deletar %s. Razão: %s' % (file_path, e))

def convertionToAVI(video, h, w):
    comando1 = f"{'ffmpeg -y -f rawvideo -vcodec rawvideo -s '}{w}{'x'}{h}{' -pix_fmt yuv420p -i D:/PesquisaAudioVisual/AudioVisualMeter/dataset/'}"
    file = video
    comando2 = ' -c:v libx264 -preset ultrafast -qp 0 D:/PesquisaAudioVisual/AudioVisualMeter/videosAVI/'
    comando = f"{comando1}{file}{'.yuv'}{comando2}{file}{'.avi'}"    
    
    run = sp.getoutput(comando)