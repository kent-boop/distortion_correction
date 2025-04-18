import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def cv_show(name,img):#显示图片
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def visualize_displacement_field(displacement_x, displacement_y, step=20):
    """
    可视化位移场（矢量图）
    :param displacement_x: x方向位移场
    :param displacement_y: y方向位移场
    :param step: 采样步长（减少显示点数量）
    :return: 位移场可视化图像
    """
    height, width = displacement_x.shape
    y, x = np.mgrid[0:height:step, 0:width:step]
    u = displacement_x[y, x]
    v = displacement_y[y, x]

    # 绘制位移场
    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='r')
    plt.title("Displacement Field")
    plt.show()

def center_crop(image, crop_width, crop_height):
    """
    对图像进行中心裁剪。

    参数:
        image: 输入图像（NumPy 数组）。
        crop_width: 裁剪区域的宽度。
        crop_height: 裁剪区域的高度。

    返回:
        裁剪后的图像。
    """
    # 获取图像的尺寸
    height, width = image.shape[:2]
    # 计算裁剪区域的起始坐标
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    # 进行中心裁剪
    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return cropped_image

#剪切畸变Shear Distortion
def Horizontal_Shear_with_displacement(image, alpha, dis_w, dis_h):
    """
    水平剪切畸变并记录位移场
    :param image: 输入图像
    :param alpha: 水平剪切因子
    :param dis_w: 输出图像宽度
    :param dis_h: 输出图像高度
    :return: 畸变图像, 位移场 (disp_x, disp_y)
    """
    h, w = image.shape[:2]
    beta = 0
    # 定义剪切矩阵
    M_shear = np.float32([[1, alpha, 0],
                    [beta, 1, 0]])

    # 计算剪切后的图像尺寸
    new_w = int(w + alpha * h)
    new_h = int(h + beta * w)
    s_x = dis_w/new_w
    s_y = dis_h/new_h
    M_total = np.float32([[s_x, alpha*s_x, 0],
                    [beta*s_y, s_y, 0]])
    sheared_image = cv.warpAffine(image, M_total, (dis_w, dis_h),borderMode=cv.BORDER_REFLECT_101)  # 进行仿射变换
    print(sheared_image.shape)
    # sheared_image = cv.resize(sheared_image, (dis_w, dis_h))

    # 计算位移场
    disp_x = np.zeros((dis_h, dis_w), dtype=np.float32)
    disp_y = np.zeros((dis_h, dis_w), dtype=np.float32)

    # 对输出图像的每个像素，计算其在原图中的坐标
    inv_M = cv.invertAffineTransform(M_total)  # 计算逆变换矩阵

    x_grid, y_grid = np.meshgrid(np.arange(dis_w), np.arange(dis_h))
    src_x = inv_M[0, 0] * x_grid + inv_M[0, 1] * y_grid + inv_M[0, 2]
    src_y = inv_M[1, 0] * x_grid + inv_M[1, 1] * y_grid + inv_M[1, 2]

    # # 限制坐标在原图范围内
    # src_x = np.clip(src_x, 0, w - 1)
    # src_y = np.clip(src_y, 0, h - 1)

    # 计算位移场
    disp_x =  (x_grid - src_x)
    disp_y =  (y_grid - src_y)

    return sheared_image, disp_x, disp_y


def Vertical_Shear_with_displacement(image, beta, dis_w, dis_h):
    """
    垂直剪切畸变并记录位移场
    :param image: 输入图像
    :param beta: 垂直剪切因子
    :param dis_w: 输出图像宽度
    :param dis_h: 输出图像高度
    :return: 畸变图像, 位移场 (disp_x, disp_y)
    """
    h, w = image.shape[:2]
    alpha = 0
    # 定义剪切矩阵
    M_shear = np.float32([[1, alpha, 0],
                    [beta, 1, 0]])

    # 计算剪切后的图像尺寸
    new_w = int(w + alpha * h)
    new_h = int(h + beta * w)
    s_x = dis_w/new_w
    s_y = dis_h/new_h
    M_total = np.float32([[s_x, alpha*s_x, 0],
                    [beta*s_y, s_y, 0]])
    sheared_image = cv.warpAffine(image, M_total, (dis_w, dis_h),borderMode=cv.BORDER_REFLECT_101)  # 进行仿射变换
    print(sheared_image.shape)
    # sheared_image = cv.resize(sheared_image, (dis_w, dis_h))

    # 计算位移场
    disp_x = np.zeros((dis_h, dis_w), dtype=np.float32)
    disp_y = np.zeros((dis_h, dis_w), dtype=np.float32)

    # 对输出图像的每个像素，计算其在原图中的坐标
    inv_M = cv.invertAffineTransform(M_total)  # 计算逆变换矩阵

    x_grid, y_grid = np.meshgrid(np.arange(dis_w), np.arange(dis_h))
    src_x = inv_M[0, 0] * x_grid + inv_M[0, 1] * y_grid + inv_M[0, 2]
    src_y = inv_M[1, 0] * x_grid + inv_M[1, 1] * y_grid + inv_M[1, 2]

    # # 限制坐标在原图范围内
    # src_x = np.clip(src_x, 0, w - 1)
    # src_y = np.clip(src_y, 0, h - 1)

    # 计算位移场
    disp_x =  (x_grid - src_x)
    disp_y =  (y_grid - src_y)

    return sheared_image, disp_x, disp_y


def Combined_Shear_with_displacement(image, alpha, beta, dis_w, dis_h):
    """
    混合剪切畸变并记录位移场
    :param image: 输入图像
    :param alpha: 水平剪切因子
    :param beta: 垂直剪切因子
    :param dis_w: 输出图像宽度
    :param dis_h: 输出图像高度
    :return: 畸变图像, 位移场 (disp_x, disp_y)
    """
    h, w = image.shape[:2]

    # 定义剪切矩阵
    M_shear = np.float32([[1, alpha, 0],
                    [beta, 1, 0]])

    # 计算剪切后的图像尺寸
    new_w = int(w + alpha * h)
    new_h = int(h + beta * w)
    s_x = dis_w/new_w
    s_y = dis_h/new_h
    M_total = np.float32([[s_x, alpha*s_x, 0],
                    [beta*s_y, s_y, 0]])
    sheared_image = cv.warpAffine(image, M_total, (dis_w, dis_h),
                                  # borderMode=cv.BORDER_REFLECT_101
                                  )  # 进行仿射变换
    print(sheared_image.shape)
    # sheared_image = cv.resize(sheared_image, (dis_w, dis_h))

    # 计算位移场
    disp_x = np.zeros((dis_h, dis_w), dtype=np.float32)
    disp_y = np.zeros((dis_h, dis_w), dtype=np.float32)

    # 对输出图像的每个像素，计算其在原图中的坐标
    inv_M = cv.invertAffineTransform(M_total)  # 计算逆变换矩阵

    x_grid, y_grid = np.meshgrid(np.arange(dis_w), np.arange(dis_h))
    src_x = inv_M[0, 0] * x_grid + inv_M[0, 1] * y_grid + inv_M[0, 2]
    src_y = inv_M[1, 0] * x_grid + inv_M[1, 1] * y_grid + inv_M[1, 2]

    # # 限制坐标在原图范围内
    # src_x = np.clip(src_x, 0, w - 1)
    # src_y = np.clip(src_y, 0, h - 1)

    # 计算位移场
    disp_x =  (x_grid - src_x)
    disp_y =  (y_grid - src_y)

    return sheared_image, disp_x, disp_y

def correct_distortion(distorted_image, disp_x, disp_y):
    h, w = distorted_image.shape[:2]

    # 生成校正后的网格
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    src_x = x_grid + disp_x
    src_y = y_grid + disp_y

    # 使用 OpenCV 的 remap 函数进行双线性插值
    corrected_image = cv.remap(distorted_image, src_x.astype(np.float32), src_y.astype(np.float32), cv.INTER_CUBIC,borderMode=cv.BORDER_REFLECT_101)

    return corrected_image



if __name__ == "__main__":
    # 读取图像
    image = cv.imread("E:\photo\chess board.jpg")
    h, w = image.shape[:2]

    # # 水平剪切
    # distorted_img, disp_x, disp_y = Horizontal_Shear_with_displacement(image, 0.1, w, h)
    # cv_show("Horizontal Sheared Image", distorted_img)
    #
    # # 垂直剪切
    # distorted_img, disp_x, disp_y = Vertical_Shear_with_displacement(image, 0.1, w, h)
    # cv_show("Vertical Sheared Image", distorted_img)

    # 混合剪切
    distorted_img, disp_x, disp_y = Combined_Shear_with_displacement(image, 0.08, 0.08, w, h)
    cv_show("Combined Sheared Image", distorted_img)

    # 可视化位移场
    visualize_displacement_field(disp_x,disp_y)
    # 校正图像
    corrected_image = correct_distortion(distorted_img, disp_x, disp_y)
    # 显示校正后的图像
    cv_show("Combined Sheared Image", distorted_img)
    cv_show("Corrected Image", corrected_image)
