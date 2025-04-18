import cv2 as cv
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

def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def normalize_coordinates(h, w):
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    return np.meshgrid(x, y)


def barrel_distortion_with_padding(image, k1, k2, k3, pad_ratio=0.5):
    """改进的桶形畸变函数（含边界扩展）"""
    h, w = image.shape[:2]
    pad_size = int(max(h, w) * pad_ratio)

    # 扩展原图边界
    padded_img = cv.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size,
                                    cv.BORDER_REFLECT_101)
    # cv_show('padded_img',padded_img)
    # 在扩展后的图像上计算位移场
    h_pad, w_pad = padded_img.shape[:2]
    x_norm, y_norm = normalize_coordinates(h_pad, w_pad)
    r = np.sqrt(x_norm ** 2 + y_norm ** 2)

    # 三阶畸变模型
    r_distorted = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)

    mask = (r != 0)
    x_distorted = np.zeros_like(x_norm)
    y_distorted = np.zeros_like(y_norm)
    x_distorted[mask] = (x_norm[mask] / r[mask]) * r_distorted[mask]
    y_distorted[mask] = (y_norm[mask] / r[mask]) * r_distorted[mask]

    # 转换为像素坐标
    map_x = ((x_distorted + 1) * w_pad) / 2
    map_y = ((y_distorted + 1) * h_pad) / 2

    # 生成位移场（基于扩展图像）
    x_grid_pad, y_grid_pad = np.meshgrid(np.arange(w_pad), np.arange(h_pad))
    displacement_x_pad = map_x - x_grid_pad
    displacement_y_pad = map_y - y_grid_pad

    # 应用畸变并裁剪
    distorted_padded = cv.remap(padded_img,
                                 map_x.astype(np.float32),
                                 map_y.astype(np.float32),
                                 cv.INTER_CUBIC,
                                # borderMode=cv.BORDER_REFLECT_101
                                )

    # 裁剪回原始尺寸
    distorted_img = distorted_padded[pad_size:-pad_size, pad_size:-pad_size]
    displacement_x = displacement_x_pad[pad_size:-pad_size, pad_size:-pad_size]
    displacement_y = displacement_y_pad[pad_size:-pad_size, pad_size:-pad_size]

    return distorted_img, displacement_x.astype(np.float32), displacement_y.astype(np.float32)


def correct_distortion(distorted_img, disp_x, disp_y):
    """改进的校正函数"""
    h, w = distorted_img.shape[:2]

    # 生成匹配的坐标网格
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    # 逆向映射坐标
    src_x = x_grid - disp_x
    src_y = y_grid - disp_y

    # 应用校正（扩展边界处理）
    corrected = cv.remap(distorted_img,
                          src_x.astype(np.float32),
                          src_y.astype(np.float32),
                          interpolation=cv.INTER_LANCZOS4,
                          borderMode=cv.BORDER_REFLECT_101)

    return corrected


# 使用示例
if __name__ == "__main__":
    # 参数设置
    k1, k2, k3 = 0.3, 0.1, 0.05
    image = cv.imread("E:\photo\chess board.jpg")
    image = cv.resize(image,[512,512])
    # 生成畸变图像和位移场
    distorted_img, disp_x, disp_y = barrel_distortion_with_padding(image, k1, k2, k3)
    visualize_displacement_field(disp_x,disp_y)
    # 验证尺寸一致性
    print(f"畸变图像尺寸: {distorted_img.shape}")
    print(f"位移场尺寸: {disp_x.shape}")

    # 执行校正
    corrected_img = correct_distortion(distorted_img, disp_x, disp_y)

    # 显示结果
    cv_show("Original", image)
    cv_show("Distorted", distorted_img)
    cv_show("Corrected", corrected_img)


