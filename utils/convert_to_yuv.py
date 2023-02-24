import os
import subprocess as sp

videos_dir = '/mnt/d/GPDS/VQA-ODV/Group1/Dist/convert'

video_list = os.listdir(videos_dir)

for video in video_list:
    print('Video atual:', video)
    command_line = f"{'ffmpeg -i '}{videos_dir}{video}{' -c:v rawvideo -pix_fmt yuv420p '}{videos_dir}{video[:-4]}{'.yuv'}"
    run = sp.getoutput(command_line)