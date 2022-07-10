import os, shutil
import subprocess as sp
import json

def setup_starting_file():
    data = { "Nome do dataset" : "NaN", "Formato dos videos" : "NaN", "Metricas" : ["NaN"], "Endereco da referencia" : "NaN", "Endereco do destino" : "NaN"}
    json_file = open("json_file.json", 'w', encoding='utf-8')

    json_data = json.dumps(data)
    json_file.write(json_data)
    json_file.close()

def read_json():
    json_file = open("json_file.json", encoding='utf-8')
    json_data = json.loads(json_file.read())
    json_file.close()
    return json_data

def edit_json():
    json_file = open("json_file.json", "w", encoding='utf-8')

    data = dict()
    data["Nome do dataset"] = input("Escreva o nome do dataset:\n")
    data["Formato dos videos"] = input("Escreva o formato dos videos:\n")
    data["Metricas"] = input("Escreva as métricas a serem utilizadas separadas por vírgula:\n").split(",")
    data["Endereco da referencia"] = input("Escreva o endereço (path) do vídeo de referência. Caso não haja, escreva NaN:\n")
    data["Endereco do destino"] = input("Escreva o endereço (path) dos vídeos a serem avaliados.\n")

    # Mais tarde, fazer função de validação para garantir que as variáveis foram inicializadas corretamente.

    json_data = json.dumps(data)
    json_file.write(json_data)
    json_file.close()

def is_json():
    try:
        open("json_file.json", encoding='utf-8').close()
        return True
    except:
        return False

def initialize(edit = False):

    if(edit == False):
        answer = "Y"
    else:
        answer = "NaN"

    if is_json():
        print("\nO arquivo existente segue o seguinte formato:")
        print(read_json())

        while (answer != "Y" and answer != "N"):
            answer = input("\nDeseja continuar com ele? (Y/N)\n")

        if answer == "N":
            setup_starting_file()
            edit_json()

    else:
        if(edit == False):
            print("\nNão há arquivo json existente, será necessário criar um.")

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