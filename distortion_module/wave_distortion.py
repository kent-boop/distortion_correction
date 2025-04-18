import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def Horizontal_Wave_Distortion(A, lambda_, image):
    h, w = image.shape[:2]
    pad_size = int(0.2 * max(h, w))
    image_padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT_101)
    h_pad, w_pad = image_padded.shape[:2]

    # 生成二维坐标网格（确保 map_x 和 map_y 尺寸一致）
    y_coords, x_coords = np.mgrid[0:h_pad, 0:w_pad]  # 关键修改：使用 np.mgrid 生成二维网格
    map_x = x_coords + A * np.sin(2 * np.pi * y_coords / lambda_)
    map_y = y_coords  # 直接使用二维网格

    # 转换为 float32
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 应用畸变
    wave_image_padded = cv2.remap(
        image_padded,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    wave_image = wave_image_padded[pad_size:-pad_size, pad_size:-pad_size]

    # 计算位移场（调整到原图坐标系）
    x_grid_orig, y_grid_orig = np.meshgrid(np.arange(w), np.arange(h))
    x_grid_pad = x_grid_orig + pad_size
    y_grid_pad = y_grid_orig + pad_size
    src_x_pad = map_x[y_grid_pad, x_grid_pad]
    src_y_pad = map_y[y_grid_pad, x_grid_pad]
    displacement_x = src_x_pad - x_grid_pad
    displacement_y = src_y_pad - y_grid_pad

    return wave_image, displacement_x, displacement_y

def Vertical_Wave_Distortion(A,lambda_,image):
    h, w = image.shape[:2]
    pad_size = int(0.2 * max(h, w))
    image_padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT_101)
    h_pad, w_pad = image_padded.shape[:2]

    # 生成二维坐标网格（确保 map_x 和 map_y 尺寸一致）
    y_coords, x_coords = np.mgrid[0:h_pad, 0:w_pad]  # 关键修改：使用 np.mgrid 生成二维网格
    map_y = y_coords + A * np.sin(2 * np.pi * x_coords / lambda_)
    map_x = x_coords  # 直接使用二维网格

    # 转换为 float32
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 应用畸变
    wave_image_padded = cv2.remap(
        image_padded,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    wave_image = wave_image_padded[pad_size:-pad_size, pad_size:-pad_size]

    # 计算位移场（调整到原图坐标系）
    x_grid_orig, y_grid_orig = np.meshgrid(np.arange(w), np.arange(h))
    x_grid_pad = x_grid_orig + pad_size
    y_grid_pad = y_grid_orig + pad_size
    src_x_pad = map_x[y_grid_pad, x_grid_pad]
    src_y_pad = map_y[y_grid_pad, x_grid_pad]
    displacement_x = src_x_pad - x_grid_pad
    displacement_y = src_y_pad - y_grid_pad

    return wave_image, displacement_x, displacement_y

def Combined_Wave_Distortion(A_h,A_v,lambda_h,lambda_v,image):
    h, w = image.shape[:2]
    pad_size = int(0.2 * max(h, w))
    image_padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT_101)
    h_pad, w_pad = image_padded.shape[:2]

    # 生成二维坐标网格（确保 map_x 和 map_y 尺寸一致）
    y_coords, x_coords = np.mgrid[0:h_pad, 0:w_pad]  # 关键修改：使用 np.mgrid 生成二维网格
    map_x = x_coords + A_h * np.sin(2 * np.pi * y_coords / lambda_h)
    map_y = y_coords + A_v * np.sin(2 * np.pi * x_coords / lambda_v)

    # 转换为 float32
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 应用畸变
    wave_image_padded = cv2.remap(
        image_padded,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        # borderMode=cv2.BORDER_REFLECT_101
    )
    wave_image = wave_image_padded[pad_size:-pad_size, pad_size:-pad_size]

    # 计算位移场（调整到原图坐标系）
    x_grid_orig, y_grid_orig = np.meshgrid(np.arange(w), np.arange(h))
    x_grid_pad = x_grid_orig + pad_size
    y_grid_pad = y_grid_orig + pad_size
    src_x_pad = map_x[y_grid_pad, x_grid_pad]
    src_y_pad = map_y[y_grid_pad, x_grid_pad]
    displacement_x = src_x_pad - x_grid_pad
    displacement_y = src_y_pad - y_grid_pad

    return wave_image, displacement_x, displacement_y

def correct_wave_distortion(distorted_image, disp_x, disp_y):
    h, w = distorted_image.shape[:2]
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    # 逆向映射坐标
    src_x = x_grid - disp_x
    src_y = y_grid - disp_y
    src_x = src_x.astype(np.float32)
    src_y = src_y.astype(np.float32)

    # 使用 Lanczos 插值和镜像边界
    corrected_image = cv2.remap(
        distorted_image,
        src_x,
        src_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return corrected_image

# 读取图像
image = cv2.imread("E:/photo/chess board.jpg")

# 定义波浪参数
A = 5  # 振幅
lambda_ = 100  # 波长

# 水平波浪畸变
# wave_image, disp_x_h, disp_y_h = Horizontal_Wave_Distortion(A, lambda_, image)
# # 显示结果
# cv2.imshow("Horizontal Wave Distortion", wave_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# visualize_displacement_field(disp_x_h,disp_y_h)
#
# # 垂直波浪畸变
# wave_image, disp_x_v, disp_y_v = Vertical_Wave_Distortion(A, lambda_, image)
# # 显示结果
# cv2.imshow("Vertical Wave Distortion", wave_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# visualize_displacement_field(disp_x_v,disp_y_v)

# 混合波浪畸变
# 定义波浪参数
A_h = 7  # 水平振幅
A_v = 7  # 垂直振幅
lambda_h = 1200  # 水平波长
lambda_v = 1200  # 垂直波长

wave_image, disp_x_c, disp_y_c = Combined_Wave_Distortion(A_h, A_v, lambda_h, lambda_v, image)
correct_img = correct_wave_distortion(wave_image,disp_x_c,disp_y_c)
# 显示结果
cv2.imshow("orin", image)
cv2.imshow("Combined Wave Distortion", wave_image)
cv2.imshow("correct_img", correct_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
visualize_displacement_field(disp_x_c,disp_y_c)


