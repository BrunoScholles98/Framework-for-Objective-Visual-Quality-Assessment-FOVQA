import os
import subprocess as sp

videos_dir = '/mnt/d/GPDS/VQA-ODV/Group1/Ref'

video_list = os.listdir(videos_dir)

for video in video_list:
    print(video)