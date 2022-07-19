import os, shutil
import subprocess as sp
import json

# create the json file
def setup_starting_file():
    data = { "Dataset name" : "NaN", "Videos file format" : "NaN", "Metrics" : ["NaN"], "Path to reference foulder" : "NaN", "Path to distorted folder" : "NaN"}
    json_file = open("parameters.json", 'w', encoding='utf-8')

    json_data = json.dumps(data)
    json_file.write(json_data)
    json_file.close()

# get content from json file
def read_json():
    json_file = open("parameters.json", encoding='utf-8')
    json_data = json.loads(json_file.read())
    json_file.close()
    return json_data

# change the json file content
def edit_json():
    json_file = open("parameters.json", "w", encoding='utf-8')

    data = dict()

    dataset_name = input("Dataset name:\n")
    if dataset_name.find('.csv') != -1:
        dataset_name = dataset_name[:dataset_name.find('.csv')]                        # excludes the '.csv' part if present
    data["Dataset name"] = dataset_name


    vid_format = input("Videos file format:\n")
    if vid_format[0] == '.':
        vid_format = vid_format[1:]                                                    # excludes the '.' if present
    vid_format = vid_format.lower()
    data["Videos file format"] = vid_format

    metrics = input("Metrics to use (separate them using a comma):\n").split(",")
    for m in len(metrics):
        metrics[m] = metrics[m].strip(' ')                                             # remove blank spaces
        metrics[m] = metrics[m].lower
    data["Metrics"] = metrics

    data["Path to reference foulder"] = input("Path to reference foulder. If there isn't one, type NaN:\n")
    data["Path to distorted folder"] = input("Path to distorted foulder:\n")

    json_data = json.dumps(data)
    json_file.write(json_data)
    json_file.close()

# tests if the necessary json file exists
def is_json():
    try:
        open("parameters.json", encoding='utf-8').close()
        return True
    except:
        return False

# open the json file
def initialize(edit = False):

    if(edit == False):
        answer = "Y"
    else:
        answer = "NaN"

    if is_json():
        print("\nThe existing file has the following format:")
        print(read_json())

        while (answer != "Y" and answer != "N"):
            answer = input("\nProceed with it? (Y/N)\n")

        if answer == "N":
            setup_starting_file()
            edit_json()

    else:
        if(edit == False):
            print("\nThere is no existing json file, let's create one.")

        setup_starting_file()
        edit_json()

    return read_json()

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

def convertionToAVI(video, h, w, path):
    comando1 = f"{'ffmpeg -y -f rawvideo -vcodec rawvideo -s '}{w}{'x'}{h}{' -pix_fmt yuv420p -i '}{path}"
    file = video
    comando2 = ' -c:v libx264 -preset ultrafast -qp 0 /home/brunoscholles/Framework/AudioVisualMeter/videosAVI/'
    comando = f"{comando1}{file}{'.yuv'}{comando2}{file}{'.avi'}"    
    
    run = sp.getoutput(comando)