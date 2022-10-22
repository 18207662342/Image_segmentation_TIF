# Image_segmentation_TIF
TIF切割
把约16000*16000分辨率的tif文件转换成1024*1024tif文件，
识别后生成labels文件
对labels文件进行数据转换成秧苗的像素坐标，
再通过像素坐标得到gps坐标，
通过把像素坐标画在tif大图上显示出来。


gps_view.py #把gps坐标显示再Google地图上，生成html文件
image_segmentation.py #把高分辨率的大图切割成小分辨率的小
pixels_gps #把labels文件转换成像素坐标，并显示在大图tif上；通过像素坐标得到gps坐标； 
