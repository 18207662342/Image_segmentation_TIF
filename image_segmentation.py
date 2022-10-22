# The path can also be read from a config file, etc.
OPENSLIDE_PATH = "C:/czg/602/app/pycharm/ANACONDA/envs/pytorch/Lib/site-packages/openslide/openslide-win64-20220811/bin"

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

# import openslide
import numpy as np
import imageio  # 用于保存瓦片
# os.add_dll_directory("path_to_working_dlls_directoy")

def train():
    slide = openslide.OpenSlide("C:/czg/602/project/Uav path planning/1.code/pycharm code/Image_segmentation/big/result.tif")
    dst_path = 'C:/czg/602/project/Uav path planning/1.code/pycharm code/Image_segmentation/input/photo_small9/'

    [m, n] = slide.dimensions  # 得出高倍下的（宽，高）
    print(m, n)
    M = 128
    N = 128
    ml = M * m // M
    nl = N * n // N

    for i in range(0, ml, M):  # 这里由于只是想实现剪切功能，暂时忽略边缘不足N*N的部分
        for j in range(0, nl, N):
            im = np.array(slide.read_region((i, j), 0, (M, N)))
            imageio.imwrite(dst_path + str(i) + '-' + str(j) + '.tif', im)  # patch命名为‘x-y’

    slide.close()  # 关闭文件

if __name__ == '__main__':
    train()
