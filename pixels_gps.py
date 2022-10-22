import matplotlib.pylab as plt
from osgeo import gdal
import cv2
import  numpy as np
import os
import gps_view as gps
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def getdata(data_loc,m,n,w,h):
    # width,height = 5472,3648    #5472*3648
    width = n
    height = m
    class_list = []
    x_min_list = []
    y_min_list = []
    x_max_list = []
    y_max_list = []
    x_center_list = []
    y_center_list = []
    with open(data_loc, "r") as f:
        for i in f.readlines():
            data_i = i.split(" ")
            class_i = str(data_i[0][0:])
            u_center_i = float(data_i[1][0:])
            v_center_i = float(data_i[2][0:])
            v_width_i = float(data_i[3][0:])
            u_height_i = float(data_i[4][0:])
            class_list.append(class_i)
            x_min_list.append(int((float(u_center_i) - 0.5 * float(v_width_i)) * width))
            y_min_list.append(int((float(v_center_i) - 0.5 * float(u_height_i)) * height))
            x_max_list.append(int((float(u_center_i) + 0.5 * float(v_width_i)) * width))
            y_max_list.append(int((float(v_center_i) + 0.5 * float(u_height_i)) * height))
            x_center_list.append(int((float(u_center_i))*width)+w)
            y_center_list.append(int((float(v_center_i))*height)+h)
        # print(len(class_list))

    return  x_center_list, y_center_list

def get_rice_objection(txt_data_dir, jpg_data_dir, m, n,w,h):
    #打开txt文件，获取秧苗位置点
    x_center_list, y_center_list = getdata(txt_data_dir, m, n,w,h)

    # 质心显示在图片上
    # img = cv2.imread(jpg_data_dir)
    # for i in range(1,len(x_center_list)):
    #     cv2.circle(img, (x_center_list[i]-w, y_center_list[i]-h), 2, (0, 0, 255), -1)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(jpg_data_dir + '.png', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # 平面坐标转换 uv-xyz

    img_points = []

    for i in range(len(x_center_list)):
        img_points.append([x_center_list[i], y_center_list[i]])
    # print(img_points)
    # img_points = np.asmatrix(img_points).T
    # img_points = np.array(img_points)
    # img_points = list(img_points)
    # print(type(img_points))
    return img_points

def pixel_to_world(camera_intrinsics, r, t, img_points):
    K_inv = camera_intrinsics.I
    R_inv = np.asmatrix(r).I
    world_points = []
    scale_world_all = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)
        cam_R_inv = np.dot(R_inv, cam_point)

        # scale = camera_parameter["h"]
        scale = t[2]
        scale_world = np.multiply(scale, cam_R_inv)
        scale_world_all.append(scale_world)
        world_point = np.asmatrix(scale_world) + np.asmatrix(t)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0] = world_point[0]
        pt[1] = world_point[1]
        pt[2] = t[2]
        world_points.append(pt.T.tolist())

    scale_world_x = []
    scale_world_y = []
    for i , j in enumerate(scale_world_all):
        scale_world_x.append(j[0])
        scale_world_y.append(j[1])

    # 画坐标散点图
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Result Analysis')
    ax1.set_xlabel('scale_world_x')
    ax1.set_ylabel('scale_world_y')
    ax1.scatter(scale_world_x, scale_world_y, c='r', marker='.')
    # 画直线图
    # ax1.plot(x2, y2, c='b', ls='--')
    plt.xlim(xmax=8, xmin=-8)
    plt.ylim(ymax=6, ymin=-6)
    plt.legend('rice')
    plt.show()

    return world_points

def read_gps(big_jpg_data_dir, img_points):
    filePath = big_jpg_data_dir  # tif文件路径
    dataset = gdal.Open(filePath)  # 打开tif

    # 质心显示在图片上
    img = cv2.imread(big_jpg_data_dir)
    for i in range(1,len(img_points)):
        cv2.circle(img, (img_points[i][0], img_points[i][1]), 8, (0, 0, 255), -1)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(os.path.dirname(big_jpg_data_dir),
                                'result_gps-128.tif'), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # cv2.imwrite('1.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


    adfGeoTransform = dataset.GetGeoTransform()  # 读取地理信息

    # 左上角地理坐标
    print('最小经度', adfGeoTransform[0])
    print('最小纬度',adfGeoTransform[3])

    nXSize = dataset.RasterXSize  # 列数
    nYSize = dataset.RasterYSize  # 行数
    print("图片高度",nXSize, "图片宽度",nYSize)

    # Xmax = adfGeoTransform[0] + 15608 * adfGeoTransform[1] + nYSize * adfGeoTransform[2] #adfGeoTransform[2]=0
    # Ymax = adfGeoTransform[3] + nXSize * adfGeoTransform[4] + 18454 * adfGeoTransform[5] #adfGeoTransform[4]=0
    # print(Xmax, Ymax)
    # x = np.linspace(adfGeoTransform[0], Xmax, nYSize)
    # print(x)
    gps = []
    longitude = []
    latitude = []
    for i in img_points:
        longitude.append(adfGeoTransform[0] + i[0] * adfGeoTransform[1] + 0 * adfGeoTransform[2])
        latitude.append(adfGeoTransform[3] + 0 * adfGeoTransform[4] + i[1] * adfGeoTransform[5])
    for i in range(len(longitude)):
        gps.append([latitude[i], longitude[i]])
    # gps = [longitude, latitude]
    # gps = np.asmatrix(gps).T
    # gps = np.array(gps)
    # print(gps)
    return gps

def draw_picture(all_gps):
    scale_world_x = []
    scale_world_y = []
    for i in all_gps:
        scale_world_x.append(i[0])
        scale_world_y.append(i[1])

    # 画坐标散点图
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Result Analysis')
    ax1.set_xlabel('scale_world_x')
    ax1.set_ylabel('scale_world_y')
    ax1.scatter(scale_world_x, scale_world_y, c='r', marker='.')
    # 画直线图
    # ax1.plot(x2, y2, c='b', ls='--')
    # plt.xlim(xmax=8, xmin=-8)
    # plt.ylim(ymax=6, ymin=-6)
    plt.legend('rice')
    plt.savefig(os.path.join(os.path.dirname(txt_data_dir), 'result'), bbox_inches='tight', pad_inches=0, dpi=600)

    plt.show()

def read_name(image_name):
    # print(image_name)
    image_name = image_name[:len(image_name) - 4]
    data_i = image_name.split("-")
    image_name_w = int(data_i[0][0:])
    image_name_h = int(data_i[1][0:])
    return image_name_w, image_name_h

if __name__ == '__main__':
    big_jpg_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'big/result.tif')
    jpg_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'input\photo_small9')  # 提取照片信息,      #输入
    txt_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'input\labels128')  # 提取秧苗位置信息   #位置更改
    html_data_dir= os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'output\html')  # 提取秧苗位置信息   #位置更改
    os.chdir(txt_data_dir)
    all_points = []
    all_gps = []
    for image_name in os.listdir(os.getcwd()):
        print(image_name)
        img = cv2.imread(os.path.join(jpg_data_dir, image_name[:len(image_name)-4])+'.tif', 1)
        m = img.shape[0]    #高度
        n = img.shape[1]    #宽度
        w, h = read_name(image_name)
        img_points = get_rice_objection(os.path.join(txt_data_dir, image_name), os.path.join(jpg_data_dir, image_name[:len(image_name)-4])+'.tif', m, n, w, h)
        all_points += img_points
    gps_data = read_gps(big_jpg_data_dir, all_points)

    # gps.draw_gps(os.path.join(html_data_dir, image_name[:len(image_name)-4]+'.HTML'), gps_data, 'red', 'orange')



