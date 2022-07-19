import pandas as pd
import sys
import math
import metrics
import metrics_nd
import json
from utils import tools

def main():

    if len(sys.argv) > 1:
        if sys.argv[1] == '-edit':
            json_data = tools.initialize(True)
    else:
        json_data = tools.initialize()


    error = False 
        
    fileName = json_data["Dataset name"]
    vid_format = json_data["Videos file format"]
    refpath = json_data["Path to reference foulder"]
    distpath = json_data["Path to distorted folder"]
    metrics_list = json_data["Metrics"]
    
    print(f"{'Dataset: '}{fileName}")
    print(f"{'Videos file format: '}{vid_format}")
    print(f"{'Reference Path: '}{refpath}")
    print(f"{'Distorded Path: '}{distpath}")
    print(f"{'Metrics: '}{metrics_list}")
    
    if fileName == "NaN" or vid_format == "NaN" or distpath == "NaN" or metrics_list == "NaN":
        print("Missing information, please retype the command adding \"-edit\" at the end to add the information")
        return
    
    if not metrics_list:                                                                                
        error = True
    elif 'all' in metrics_list and len(metrics_list) > 1:                                                   
        error = True
    elif 'all' in metrics_list and len(metrics_list) == 1:
        metrics_list.remove('all')
        metrics_list = ['ssim','msssim','psnr','mse','vmaf','rmse','snr','wsnr','uqi','pbvif','niqe']
    
    if not error:
            
        df = pd.read_csv(fileName, sep = ';')
        
        tools.cleanFrameFolder()

        # nd means "no dimensions", the dimensions are really only necessary for videos in the .yuv format, other formats do not need the dimensions   #
        # nd functions do not take the dimensions as parameters, that's why we separate them                                                           #
                                                                                                                                                       #
        # No-Reference metrics do not use the reference path as parameter, that's why we separate them                                                 # 
                                                                                                                                                       #        
        refKeys = {'ssim': metrics.measureSSIM, 'msssim': metrics.measureMSSSIM, 'psnr': metrics.measurePSNR,                                          #
        'mse': metrics.measureMSE, 'vmaf': metrics.measureVMAF, 'rmse': metrics.measureRMSE, 'snr': metrics.measureSNR,                                #
        'wsnr': metrics.measureWSNR, 'uqi': metrics.measureUQI, 'pbvif': metrics.measurePBVIF}                                                         #
        norefKeys = {'niqe': metrics.measureNIQE}                                                                                                      #
                                                                                                                                                       #
        refKeys_nd = {'ssim': metrics_nd.measureSSIM, 'msssim': metrics_nd.measureMSSSIM, 'psnr': metrics_nd.measurePSNR,                              #
        'mse': metrics_nd.measureMSE, 'vmaf': metrics_nd.measureVMAF, 'rmse': metrics_nd.measureRMSE, 'snr': metrics_nd.measureSNR,                    #
        'wsnr': metrics_nd.measureWSNR, 'uqi': metrics_nd.measureUQI, 'pbvif': metrics_nd.measurePBVIF}                                                #
        norefKeys_nd = {'niqe': metrics_nd.measureNIQE}                                                                                                #
        
        for i in metrics_list:                                                   # If the metric column is missing, create it
            if i not in df.keys():
                df.insert(len(df.iloc[0]), i, math.nan)
                
        isyuv = False                                                     
        
        if vid_format == 'yuv':
            if 'height' not in df.keys() or 'width' not in df.keys():
                print("Csv error, missing dimension columns for yuv files...")
                return
            else:
                isyuv = True

                
        for i in range(len(df)):
            for j in range(len(metrics_list)):
                if metrics_list[j] in refKeys or metrics_list[j] in refKeys_nd:
                    if pd.isna(df.loc[i, metrics_list[j]]):                       # Performs the calculation if the position in the csv is empty
                        if isyuv:
                            df.loc[i, metrics_list[j]] = refKeys[metrics_list[j]](df['refFile'].iloc[i], df['testFile'].iloc[i],df['height'].iloc[i], df['weight'].iloc[i], refpath, distpath)
                        else:
                            df.loc[i, metrics_list[j]] = refKeys_nd[metrics_list[j]](df['refFile'].iloc[i], df['testFile'].iloc[i],vid_format, refpath, distpath)
                        df.to_csv(fileName, sep = ';', index = False)
                elif metrics_list[j] in norefKeys or metrics_list[j] in norefKeys_nd:
                    if pd.isna(df.loc[i, metrics_list[j]]):                       # Performs the calculation if the position in the csv is empty
                        if isyuv:
                            df.loc[i, metrics_list[j]] = norefKeys[metrics_list[j]](df['testFile'].iloc[i],df['height'].iloc[i], df['weight'].iloc[i], distpath)
                        else:
                            df.loc[i, metrics_list[j]] = norefKeys_nd[metrics_list[j]](df['testFile'].iloc[i],vid_format, distpath)
                        df.to_csv(fileName, sep = ';', index = False)
    else:
        print("error in the metrics list")

if __name__ == "__main__":
    main()