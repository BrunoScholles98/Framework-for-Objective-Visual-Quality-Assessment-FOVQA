import pandas as pd
import math
import metricsGPDS
from utils import toolsGPDS

nomeArquivo = 'IVPL_Dataset.csv'
df = pd.read_csv(nomeArquivo, sep = ';')

toolsGPDS.startAsADM
toolsGPDS.cleanFrameFolder()

refKeys = ['ssim', 'msssim', 'psnr', 'mse', 'vmaf', 'rmse', 'snr', 'wsnr', 'uqi', 'pbvif'] #Array de métricas a serem utilizadas

refFunctions = [metricsGPDS.measureSSIM, metricsGPDS.measureMSSSIM,   # Array de funções dessas mesmas
                metricsGPDS.measurePSNR, metricsGPDS.measureMSE,      # métricas (importante deixar as refKeys
                metricsGPDS.measureVMAF, metricsGPDS.measureRMSE,     # na mesma ordem que as refFunctions)
                metricsGPDS.measureSNR,  metricsGPDS.measureWSNR,
                metricsGPDS.measureUQI,  metricsGPDS.measurePBVIF]

for i in refKeys:
    if i not in df.keys():
        df.insert(len(df.iloc[0]), i, math.nan)

for i in range(len(df)):
    for j in range(len(refKeys)):
        if pd.isna(df.loc[i, refKeys[j]]):
            df.loc[i, refKeys[j]] = refFunctions[j](df['refFile'].iloc[i], df['testFile'].iloc[i], df['height'].iloc[i], df['weight'].iloc[i])
            df.to_csv(nomeArquivo, sep = ';', index = False)

#------------[SEÇÃO DO CÓDIGO VOLTADO PARA MÉTRICAS SEM REFERÊNCIA]----------------
norefKeys = ['niqe'] #Array de métricas a serem utilizadas
norefFunctions = [metricsGPDS.measureNIQE] # Array de funções dessas mesmas métricas (importante deixar as norefKeys na mesma ordem que
                                           # as norefFunctions)
for i in norefKeys:
    if i not in df.keys():
        df.insert(len(df.iloc[0]), i, math.nan)

for i in range(len(df)):
    for j in range(len(norefKeys)):
        if pd.isna(df.loc[i, norefKeys[j]]):
            df.loc[i, norefKeys[j]] = norefFunctions[j](df['testFile'].iloc[i], df['height'].iloc[i], df['weight'].iloc[i])
            df.to_csv(nomeArquivo, sep = ';', index = False)