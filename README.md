## Contents

1. [Description](#desc)
2. [License](#lic)
3. [Requirements](#req)
4. [Instructions](#inst)
6. [Contact](#contact)
7. [References](#refs)
8. [Thanks](#thanks)

<a name="desc"></a>
# Framework for Objective Visual Quality Assessment - FOVQA

This is a Quality Assessment Framework that provides researchers with the flexibility, consistency, and scalability they need to evaluate and compare quality metrics, promoting the reproducibility of results. It currently has 11 visual quality metrics that use 3 different libraries: Scikit-video, FFmpeg toolkit, and PyMetrikz.

The framework is currently adapted to work with full-reference and no-reference metrics for measuring the quality of 2D videos. Its main functionality is to run all its 11 current metrics (or a set of them) in sequence automatically for a dataset of video. Currently the framework has the following metrics: ***ssim, msssim, psnr, mse, vmaf, rmse, snr, wsnr, uqi, pbvif, and niqe***.

It can also be used to generate statistics of the results for metrics comparison purposes. 

Reference of the scientific article:
- Saigg, C. L., Dias, B. S. S., Costa, A. H. M., Farias, M. C. Q., & Martinez, H. B. (2022). A Python Framework for Objective Visual Quality Assessment. Conference on Graphics, Patterns and Images (SIBGRAPI), 35., 2022, Natal/RN. Porto Alegre: Sociedade Brasileira de Computação. pp. 105-109. DOI: https://doi.org/10.5753/sibgrapi.est.2022.23271.

The full article is available at [link](https://doi.org/10.5753/sibgrapi.est.2022.23271).

<a name="lic"></a>
## License

Still to be defined...

<a name="req"></a>
## Requirements

This section will show you the requirements for the Framework script to work correctly.

### Installation
To run the framework, you must first have the following tools installed:

##### Python Libraries (required)

- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- [Pandas](https://pandas.pydata.org/)
- [OpenCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [scikit-video](http://www.scikit-video.org/stable/)
- [PyMetrikz](https://gitlab.com/gpds-unb/pymetrikz)

##### Other Applications (required)
- [FFmpeg toolkit](https://ffmpeg.org/)
- [VMAF](https://github.com/Netflix/vmaf)

##### Recommended Tools (optional)
- [screen](https://linuxize.com/post/how-to-use-linux-screen/)

### General Organization

For the Framework to work correctly, we must follow a few rules. The first is the video path: there is no problem if the distortion and reference videos are in the same folders or in separate folders. However, all reference videos together in the same folder, just as the distortion videos must be in the same folder as well.

The second rule is that the Framework needs to read the name of all the videos, among other characteristics. Therefore, **you must create a CSV table from the Dataset containing**:

- The name of all reference videos (required).
- The name of all the distorted videos (required).
- Mos of the videos (optional).
- Video HRC.
- Type of degradation of the video (optional).
- Dimension: height of the video (required).
- Dimensions: video width (required).

Thus, we have the following example table in **which the columns must follow exactly the names defined below**:

|refFile|testFile|Mos|HRC|videoDegradationType|height|width|
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
|bus|bus_dirac_1|0.9141|TC01|dirac|1088|1920|
|bus|bus_dirac_2|1.761|TC02|dirac|1088|1920|
|bus|bus_dirac_3|2.545|TC03|dirac|1088|1920|
|bus|bus_h264_1|-0.060209|TC04|h264|1088|1920|
|bus|bus_h264_2|0.29378|TC05|h264|1088|1920|
|bus|bus_h264_3|0.42517|TC06|h264|1088|1920|
|bus|bus_mpeg2_1|0.037114|TC08|mpeg2|1088|1920|
|bus|bus_mpeg2_2|0.61435|TC09|mpeg2|1088|1920|
|bus|bus_mpeg2_3|1.5951|TC10|mpeg2|1088|1920|

Finally, if you want to generate the statistics of the metrics run in your Dataset, **you must have the columns describing the Mos and the type of degradation of the videos**.

<a name="inst"></a>
## Instructions

#### Running the Framework

After installing all the requirements, we can run our Framework. First, we recommend that you use the `screen` command, as shown above, so that the framework runs in the background. An example of the command we can use is:

`$ screen -S Framework`

So, inside the folder where the Framework is, we can run the algorithm for the first time, using the following command:

`$ python3 framework.py -edit`

Then the framework will make some requests, such as:
- Path from the Dataset CSV.
- Format of the video files to be evaluated (e.g.: yuv, avi, mp4, mkv, etc).
- The metrics to be performed. If you want to run them all, you must type `all`.
- The path to the folder containing the reference videos.
- The path to the folder containing the distorted videos.

Thus, the Framework should generate a json file, like the example below:

```json
{
    "Dataset Path":"/home/linuxbrew/Framework-for-Objective-Visual-Quality-Assessment-FOVQA/results/IVPL_Dataset_Complete.csv",
    "Videos file format":"yuv",
    "Metrics":["ssim","vmaf","snr","uqi"],
    "Path to reference folder":"/mnt/nas/Databases/VideoQuality/IVPL_VideoDataset/reference_videos/",
    "Path to distorted folder":"/mnt/nas/Databases/VideoQuality/IVPL_VideoDataset/video_sequences/"
}
```
Finally, just wait for the Framework to finish running, and it will save the results in the CSV of the Dataset. If any problems occur while the Framework is running, or you want to run the Framework on the same dataset it was run on last time, just execute:

`$ python3 framework.py`

For other datasets, path changes, etc., run the script using `-edit` again.

It is worth adding that in case there are any problems generating the json file, we recommend that you delete the **parameters.json** file and run the script again. Any errors like this should be fixed soon.

<a name="stats"></a>
#### Statistics

You can use the **statistics.py** code to calculate the correlations of each metric used with the Mean Opinion Score (Mos). The calculations are done using:

- Pearson Correlation.
- Spearman Correlation
- Kendall Correlation
- RMSE

At the end, in the "frameworkXMos" folder you will get a set of scatter plots like the one shown below, and in the "statistics" folder a CSV table will be generated with all the calculated values.

![](https://i.postimg.cc/G2k0gGgh/vmaf.png)

||pearson|spearman|kendall|RMSE|
| :------------: | :------------: | :------------: | :------------: | :------------: |
|ssim|-0.5732|-0.6418|-0.4711|0.5487|
|vmaf|-0.6077|-0.6108|-0.4619|0.5175|
|snr|-0.3889|-0.3900|-0.2967|0.4137|
|uqi|-0.1236|-0.2110|-0.1760|0.3880|


<a name="contact"></a>
## Contact

Please send all comments, questions, reports and suggestions (especially if you would like to contribute) to brunoscholles98@gmail.com or  caio.saigg@gmail.com

<a name="contrib"></a>
## Contributing

If you would like to contribute with new algorithms, increment of code performance, documentation or another kind of modifications, please contact us.

<a name="refs"></a>
## References

#### Structural Similarity Index (SSIM)
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality 
assessment: From error visibility to structural similarity" IEEE Transactions 
on Image Processing, vol. 13, no. 4, pp.600-612, Apr. 2004

#### Multi-scale SSIM Index (MSSSIM)
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality
assessment: From error visibility to structural similarity" IEEE Transactions
on Image Processing, vol. 13, no. 4, pp.600-612, Apr. 2004

#### Video Multi-Method Assessment Fusion (VMAF)
Z. Li, A. Aaron, I. Katsavounidis, A. Moorthy, and M. Manohara,
“Toward a practical perceptual video quality metric,” 2016.
[Online]. Available: 
https://medium.com/netflix-techblog/toward-a-practical-perceptualvideo-quality-metric-653f208b9652

J. Y. Lin, T.-J. Liu, E. C.-H. Wu, and C.-C. J. Kuo, “A fusion-
based video quality assessment (fvqa) index,” in Signal and Information
Processing Association Annual Summit and Conference (APSIPA), 2014
Asia-Pacific, 2014, pp. 1–5.

#### Universal Image Quality Index (UQI)  
Zhou Wang and Alan C. Bovik, "A Universal Image Quality Index", IEEE Signal
Processing Letters, 2001

#### Visual Information Fidelity (VIF)  
H. R. Sheikh and A. C. Bovik, "Image Information and Visual Quality"., IEEE
Transactions on Image Processing, (to appear).

#### Weighted Signal-to-Noise Ratio (WSNR)  
T. Mitsa and K. Varkur, "Evaluation of contrast sensitivity functions for the
formulation of quality measures incorporated in halftoning algorithms",
ICASSP '93-V, pp. 301-304.

#### Signal-to-Noise Ratio (SNR, PSNR)  
J. Mannos and D. Sakrison, "The effects of a visual fidelity criterion on the
encoding of images", IEEE Trans. Inf. Theory, IT-20(4), pp. 525-535, July 1974

<a name="thanks"></a>
## Thanks

Special thanks to [Mylene C. Q. Farias](http://www.ene.unb.br/mylene/), [Helard B. Martinez](https://people.ucd.ie/helard.becerra), and [André Henrique M. da Costa](https://www.escavador.com/sobre/277751988/andre-henrique-macedo-da-costa).
