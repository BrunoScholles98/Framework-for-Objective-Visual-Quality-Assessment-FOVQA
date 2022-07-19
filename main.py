import pandas as pd
import sys
import math
import metricsGPDS
import metricsGPDS_nd
import json
from utils import toolsGPDS

def main():

    if len(sys.argv) > 1:
        if sys.argv[1] == '-edit':
            json_data = toolsGPDS.initialize(True)
    else:
        json_data = toolsGPDS.initialize()


    erro = False 
        
    nomeArquivo = json_data["Nome do dataset"]
    formato = json_data["Formato dos videos"]
    refpath = json_data["Endereco da referencia"]
    distpath = json_data["Endereco do destino"]
    lista = json_data["Metricas"]
    
    print(f"{'Dataset: '}{nomeArquivo}")
    print(f"{'Formato: '}{formato}")
    print(f"{'Path referencia: '}{refpath}")
    print(f"{'Path distpath: '}{distpath}")
    print(f"{'Metricas: '}{lista}")
    
    if nomeArquivo == "NaN" or formato == "NaN" or refpath == "NaN" or distpath == "NaN" or lista == "NaN":
        print("Informações faltando, por favor escreva novamente o comando adicionando \"-edit\" no final para adicionar as informações")
        return
    
    if not lista:                                                                                
        erro = True
    elif 'all' in lista and len(lista) > 1:                                                    # A opção 'all' só pode vir sozinha
        erro = True
    elif 'all' in lista and len(lista) == 1:
        lista.remove('all')
        lista = ['ssim','msssim','psnr','mse','vmaf','rmse','snr','wsnr','uqi','pbvif','niqe']
    
    if not erro:
            
        nomeArquivo = f"{nomeArquivo}{'.csv'}"
        df = pd.read_csv(nomeArquivo, sep = ';')
        
        toolsGPDS.cleanFrameFolder()
        
        refKeys = {'ssim': metricsGPDS.measureSSIM, 'msssim': metricsGPDS.measureMSSSIM, 'psnr': metricsGPDS.measurePSNR,
        'mse': metricsGPDS.measureMSE, 'vmaf': metricsGPDS.measureVMAF, 'rmse': metricsGPDS.measureRMSE, 'snr': metricsGPDS.measureSNR,
        'wsnr': metricsGPDS.measureWSNR, 'uqi': metricsGPDS.measureUQI, 'pbvif': metricsGPDS.measurePBVIF}
        norefKeys = {'niqe': metricsGPDS.measureNIQE}

        refKeys_nd = {'ssim': metricsGPDS_nd.measureSSIM, 'msssim': metricsGPDS_nd.measureMSSSIM, 'psnr': metricsGPDS_nd.measurePSNR,
        'mse': metricsGPDS_nd.measureMSE, 'vmaf': metricsGPDS_nd.measureVMAF, 'rmse': metricsGPDS_nd.measureRMSE, 'snr': metricsGPDS_nd.measureSNR,
        'wsnr': metricsGPDS_nd.measureWSNR, 'uqi': metricsGPDS_nd.measureUQI, 'pbvif': metricsGPDS_nd.measurePBVIF}
        norefKeys_nd = {'niqe': metricsGPDS_nd.measureNIQE}          
        
        for i in lista:                                                   # Se a coluna referente à métrica está faltando, cria ela
            if i not in df.keys():
                df.insert(len(df.iloc[0]), i, math.nan)
                
        isyuv = False                                                     
        
        # nd é a nossa sigla para "no dimensions", as dimensões só são realmente necessárias para vídeos no formato .yuv, outros formatos não precisam das dimensões
        
        if formato == 'yuv':
            if 'height' not in df.keys() or 'width' not in df.keys():
                print("Erro no csv, colunas de dimensões faltando...")
                return
            else:
                isyuv = True

                
        for i in range(len(df)):
            for j in range(len(lista)):
                if lista[j] in refKeys or lista[j] in refKeys_nd:
                    if pd.isna(df.loc[i, lista[j]]):                       # Realiza o cálculo caso a posição no csv esteja vazia
                        if isyuv:
                            df.loc[i, lista[j]] = refKeys[lista[j]](df['refFile'].iloc[i], df['testFile'].iloc[i],df['height'].iloc[i], df['weight'].iloc[i], refpath, distpath)
                        else:
                            df.loc[i, lista[j]] = refKeys_nd[lista[j]](df['refFile'].iloc[i], df['testFile'].iloc[i],formato, refpath, distpath)
                        df.to_csv(nomeArquivo, sep = ';', index = False)
                elif lista[j] in norefKeys or lista[j] in norefKeys_nd:
                    if pd.isna(df.loc[i, lista[j]]):                       # Realiza o cálculo caso a posição no csv esteja vazia
                        if isyuv:
                            df.loc[i, lista[j]] = norefKeys[lista[j]](df['testFile'].iloc[i],df['height'].iloc[i], df['weight'].iloc[i], distpath)
                        else:
                            df.loc[i, lista[j]] = norefKeys_nd[lista[j]](df['testFile'].iloc[i],formato, distpath)
                        df.to_csv(nomeArquivo, sep = ';', index = False)
    else:
        print("Erro na lista de metricas")

if __name__ == "__main__":
    main()