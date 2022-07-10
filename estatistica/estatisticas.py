from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import statistics

def rmse(pred, target):
    return np.sqrt(((pred - target) ** 2).mean())
    
def normaliza(vetor):
    minimo = vetor.min()
    maximo = vetor.max() - minimo
    for i in range(len(vetor)):
        vetor[i] = (vetor[i] - minimo)/maximo
    return vetor

def perVideo(metrica, valores_Mos):
    valores_metrica = df[metrica].values.tolist()
    pearson, x = stats.pearsonr(valores_Mos,valores_metrica)
    spearman, y = stats.spearmanr(valores_Mos,valores_metrica)
    kendall, z = stats.kendalltau(valores_Mos,valores_metrica)
    RMSE = rmse(np.array(valores_Mos),np.array(valores_metrica))
    #metrica_normalizada = normaliza(df[metrica].values)
    fig, ax = plt.subplots()
    #ax.scatter(metrica_normalizada, Mos_normalizado, edgecolors=(0, 0, 0))
    ax.scatter(valores_metrica, valores_Mos, edgecolors=(0, 0, 0))
    #ax.plot([min(Mos_normalizado), max(Mos_normalizado)], [min(Mos_normalizado), max(Mos_normalizado)], 'k--', lw=4)
    #ax.plot([min(dados1), max(dados1)], [min(dados1), max(dados1)], 'k--', lw=4)
    ax.set_ylabel('Mos')
    ax.set_xlabel('Framework')
    nl = '\n'
    titulo = f"{metrica}"
    plt.title(titulo)
    estatisticas = f"{'pearson = {%.3f}'%pearson}{nl}{'spearman = {%.3f}'%spearman}{nl}{'kendall = {%.3f}'%kendall}{nl}{'rmse = {%.3f}'%RMSE}"
    ax.text(np.array(valores_metrica).min(), np.array(valores_Mos).min(), estatisticas, color='black', bbox=dict(facecolor='none', edgecolor='black'))
    backlash = '\\'
    plt.savefig(f"{'frameworkXMos'}{backlash}{metrica}{'.png'}")

def perDistortion(metrica, medias_Mos, desvio_Mos, distorcoes):
    medias_metrica = []
    desvio_metrica = []
    for dist in distorcoes:
        separador = df.loc[(df['videoDegradationType'] == dist)]
        valores_metrica = separador[metrica].values.tolist()
        medias_metrica.append(statistics.mean(valores_metrica))
        desvio_metrica.append(statistics.stdev(valores_metrica))
    
    pearson, x = stats.pearsonr(medias_Mos,medias_metrica)
    spearman, y = stats.spearmanr(medias_Mos,medias_metrica)
    kendall, z = stats.kendalltau(medias_Mos,medias_metrica)
    RMSE = rmse(np.array(medias_Mos),np.array(medias_metrica))
    fig, ax = plt.subplots()
    ax.errorbar(medias_metrica, medias_Mos, desvio_Mos, desvio_metrica, fmt='o', linewidth=2, capsize=6)
    ax.set_ylabel('Mos')
    ax.set_xlabel('Framework')
    nl = '\n'
    titulo = f"{metrica}"
    plt.title(titulo)
    estatisticas = f"{'pearson = {%.3f}'%pearson}{nl}{'spearman = {%.3f}'%spearman}{nl}{'kendall = {%.3f}'%kendall}{nl}{'rmse = {%.3f}'%RMSE}"
    ax.text(np.array(medias_metrica).min() - desvio_metrica[medias_metrica.index(np.array(medias_metrica).min())], np.array(medias_Mos).min() - desvio_Mos[medias_Mos.index(np.array(medias_Mos).min())], estatisticas, color='black', bbox=dict(facecolor='none', edgecolor='black'))
    for i in range(len(medias_metrica)):
        ax.text(medias_metrica[i] + 0.15*desvio_metrica[i], medias_Mos[i] + 0.15*desvio_Mos[i], distorcoes[i], color='black', bbox=dict(facecolor='none', edgecolor='red'))

    backlash = '\\'
    plt.savefig(f"{'distortionsXMos'}{backlash}{metrica}{'.png'}")
    
df = pd.read_csv('D:\\PesquisaAudioVisual\\AudioVisualMeter\\LIVE_Dataset_Complete.csv',sep=';')
lista_metricas = ['ssim','msssim','psnr','niqe','vmaf','rmse','snr','wsnr','uqi','pbvif']
distorcoes = df['videoDegradationType'].values.tolist()
distorcoes = list(set(distorcoes))

medias_Mos = []
desvio_Mos = []
for dist in distorcoes:
    separador = df.loc[(df['videoDegradationType'] == dist)]
    valores_Mos = separador['Mos'].values.tolist()
    medias_Mos.append(statistics.mean(valores_Mos))
    desvio_Mos.append(statistics.stdev(valores_Mos))

#print(desvio_Mos)
valores_Mos = df['Mos'].values.tolist()
#Mos_normalizado = normaliza(df['Mos'].values)

for metrica in lista_metricas:
   perVideo(metrica, valores_Mos)
   perDistortion(metrica, medias_Mos, desvio_Mos, distorcoes)