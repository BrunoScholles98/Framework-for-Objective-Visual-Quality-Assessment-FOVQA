from scipy import stats
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import statistics
from utils import tools

def rmse(pred, target):
    return np.sqrt(((pred - target) ** 2).mean())
    
def normalize(vector):
    minimum = vector.min()
    maximum = vector.max() - minimum
    for i in range(len(vector)):
        vector[i] = (vector[i] - minimum)/maximum
    return vector

def perVideo(metric, values_Mos, Mos_normalized, tabel_results):
    values_metric = df[metric].values.tolist()
    pearson, x = stats.pearsonr(values_Mos,values_metric)
    spearman, y = stats.spearmanr(values_Mos,values_metric)
    kendall, z = stats.kendalltau(values_Mos,values_metric)
    metric_normalized = normalize(df[metric].values)
    RMSE = rmse(np.array(Mos_normalized),np.array(metric_normalized))
    tabel_results[metric].append(pearson)
    tabel_results[metric].append(spearman)
    tabel_results[metric].append(kendall)
    tabel_results[metric].append(RMSE)
    fig, ax = plt.subplots()
    #ax.scatter(metric_normalizada, Mos_normalized, edgecolors=(0, 0, 0))
    ax.scatter(values_metric, values_Mos, edgecolors=(0, 0, 0))
    #ax.plot([min(Mos_normalized), max(Mos_normalized)], [min(Mos_normalized), max(Mos_normalized)], 'k--', lw=4)
    #ax.plot([min(dados1), max(dados1)], [min(dados1), max(dados1)], 'k--', lw=4)
    ax.set_ylabel('Mos')
    ax.set_xlabel('Predicted')
    nl = '\n'
    title = f"{metric}"
    plt.title(title)
    statistics = f"{'pearson = {%.3f}'%pearson}{nl}{'spearman = {%.3f}'%spearman}{nl}{'kendall = {%.3f}'%kendall}{nl}{'rmse = {%.3f}'%RMSE}"
    ax.text(np.array(values_metric).min(), np.array(values_Mos).min(), statistics, color='black', bbox=dict(facecolor='none', edgecolor='black'))
    slash = '/'
    plt.savefig(f"{'statistics'}{slash}{'frameworkXMos'}{slash}{metric}{'.png'}")
    return tabel_results

def perDistortion(metric, mean_Mos, deviation_Mos, distortions):
    mean_metric = []
    deviation_metric = []
    for dist in distortions:
        separator = df.loc[(df['videoDegradationType'] == dist)]
        values_metric = separator[metric].values.tolist()
        mean_metric.append(statistics.mean(values_metric))
        deviation_metric.append(statistics.stdev(values_metric))
    
    pearson, x = stats.pearsonr(mean_Mos,mean_metric)
    spearman, y = stats.spearmanr(mean_Mos,mean_metric)
    kendall, z = stats.kendalltau(mean_Mos,mean_metric)
    RMSE = rmse(np.array(mean_Mos),np.array(mean_metric))
    fig, ax = plt.subplots()
    ax.errorbar(mean_metric, mean_Mos, deviation_Mos, deviation_metric, fmt='o', linewidth=2, capsize=6)
    ax.set_ylabel('Mos')
    ax.set_xlabel('Framework')
    nl = '\n'
    titulo = f"{metric}"
    plt.title(titulo)
    statistics = f"{'pearson = {%.3f}'%pearson}{nl}{'spearman = {%.3f}'%spearman}{nl}{'kendall = {%.3f}'%kendall}{nl}{'rmse = {%.3f}'%RMSE}"
    ax.text(np.array(mean_metric).min() - deviation_metric[mean_metric.index(np.array(mean_metric).min())], np.array(mean_Mos).min() - deviation_Mos[mean_Mos.index(np.array(mean_Mos).min())], statistics, color='black', bbox=dict(facecolor='none', edgecolor='black'))
    for i in range(len(mean_metric)):
        ax.text(mean_metric[i] + 0.15*deviation_metric[i], mean_Mos[i] + 0.15*deviation_Mos[i], distortions[i], color='black', bbox=dict(facecolor='none', edgecolor='red'))

    slash = '/'
    plt.savefig(f"{'statistics'}{slash}{'distortionsXMos'}{slash}{metric}{'.png'}")
    

json_data = tools.initialize()

error = False

fileName = json_data["Dataset Path"]
metrics_list = json_data["Metrics"]

if fileName == "NaN" or metrics_list == "NaN":
    print("Missing information, please add the information in the json")
    sys.exit()

if not metrics_list:                                                                                
    error = True
elif 'all' in metrics_list and len(metrics_list) > 1:                                                   
    error = True
elif 'all' in metrics_list and len(metrics_list) == 1:
    metrics_list.remove('all')
    metrics_list = ['ssim','msssim','psnr','mse','vmaf','rmse','snr','wsnr','uqi','pbvif','niqe']

if not error:

    df = pd.read_csv(fileName,sep=';')
    if 'videoDegradationType' in df.keys():
        distortions = df['videoDegradationType'].values.tolist()
        distortions = list(set(distortions))

        mean_Mos = []
        deviation_Mos = []
        for dist in distortions:
            separator = df.loc[(df['videoDegradationType'] == dist)]
            values_Mos = separator['Mos'].values.tolist()
            mean_Mos.append(statistics.mean(values_Mos))
            deviation_Mos.append(statistics.stdev(values_Mos))

    #print(deviation_Mos)
    values_Mos = df['Mos'].values.tolist()
    Mos_normalized = normalize(df['Mos'].values)

    tabel_results = {'Correlation values': ['Pearson', 'Spearman', 'Kendall', 'RMSE']}

    for metric in metrics_list:
        tabel_results[metric] = []

    for metric in metrics_list:
        tabel_results = perVideo(metric, values_Mos, Mos_normalized, tabel_results)
        if 'videoDegradationType' in df.keys():
            perDistortion(metric, mean_Mos, deviation_Mos, distortions)


    csv = pd.DataFrame(tabel_results)
    csv.to_csv('statistics/correlation_results.csv', sep = ';', index = False)