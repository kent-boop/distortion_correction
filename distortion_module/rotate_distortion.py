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


def rotate_image_with_displacement(image, angle):
    """旋转图像并保持原始尺寸"""
    h, w = image.shape[:2]
    pad_size = int(max(h, w) * 0.5) # 扩展50%边界

    # 扩展图像边界
    padded_img = cv2.copyMakeBorder(image,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REFLECT_101)

    # 计算新的中心点
    new_center = (w // 2 + pad_size, h // 2 + pad_size)

    # 创建旋转矩阵
    M = cv2.getRotationMatrix2D(new_center, angle, 1.0)

    # 计算旋转后的完整图像
    rotated_padded = cv2.warpAffine(padded_img, M,
                                    (w + 2 * pad_size, h + 2 * pad_size),
                                    flags=cv2.INTER_CUBIC,
                                    # borderMode=cv2.BORDER_REFLECT_101
                                    )

    # 裁剪回原始尺寸
    rotated_image = rotated_padded[pad_size:-pad_size, pad_size:-pad_size]

    # 计算位移场
    inv_M = cv2.invertAffineTransform(M)
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    # 转换为扩展后坐标系
    x_pad = x_grid + pad_size
    y_pad = y_grid + pad_size

    # 计算原始坐标
    src_x = inv_M[0, 0] * x_pad + inv_M[0, 1] * y_pad + inv_M[0, 2]
    src_y = inv_M[1, 0] * x_pad + inv_M[1, 1] * y_pad + inv_M[1, 2]

    # 转换回原始坐标系
    src_x = src_x - pad_size
    src_y = src_y - pad_size

    # 生成位移场
    disp_x = src_x - x_grid
    disp_y = src_y - y_grid

    return rotated_image, disp_x.astype(np.float32), disp_y.astype(np.float32)


# def generate_random_rotation_with_displacement(image, angle_range=(-30, 30)):
#     """生成随机旋转的图像并记录位移场"""
#     angle = np.random.uniform(angle_range[0], angle_range[1])
#     rotated_image, disp_x, disp_y = rotate_image_with_displacement(image, angle)
#     return rotated_image, disp_x, disp_y
#
# def generate_random_rotation_with_displacement(image, angle_range=(-30, 30)):
#     """
#     生成随机旋转的图像并记录位移场
#     :param image: 输入图像
#     :param angle_range: 旋转角度范围（默认：-30°到30°）
#     :return: 旋转后的图像, 位移场 (disp_x, disp_y)
#     """
#     # 生成随机旋转角度
#     angle = np.random.uniform(angle_range[0], angle_range[1])
#
#     # 应用旋转并记录位移场
#     rotated_image, disp_x, disp_y = rotate_image_with_displacement(image, angle)
#
#     return rotated_image, disp_x, disp_y

def correct_distortion(rotated_image, disp_x, disp_y):
    h, w = rotated_image.shape[:2]
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    reverse_map_x = x_grid - disp_x
    reverse_map_y = y_grid - disp_y
    corrected_image = cv2.remap(rotated_image, reverse_map_x.astype(np.float32),
                                reverse_map_y.astype(np.float32), cv2.INTER_CUBIC)
    return corrected_image


if __name__ == "__main__":
    # 读取图像
    image = cv2.imread("E:\photo\dog.png")
    image = cv2.resize(image,[512,512])
    rotated_img, disp_x, disp_y = rotate_image_with_displacement(image, 30)
    corrected_img = correct_distortion(rotated_img, disp_x, disp_y)

    print("原图尺寸:", image.shape)
    print("旋转图像尺寸:", rotated_img.shape)

    cv2.imshow("Original", image)
    cv2.imshow("Rotated", rotated_img)
    cv2.imshow("Corrected", corrected_img)
    cv2.waitKey(0)

