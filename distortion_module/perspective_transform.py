import cv2
import numpy as np
from matplotlib import pyplot as plt

def visualize_displacement_field(displacement_x, displacement_y, step=20):
    height, width = displacement_x.shape
    y, x = np.mgrid[0:height:step, 0:width:step]
    u = displacement_x[y, x]
    v = displacement_y[y, x]
    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='r')
    plt.title("Displacement Field")
    plt.show()

# def visualize_displacement_field(displacement_x, displacement_y, step=20):
#     """优化后的位移场可视化（带自动归一化）"""
#     y, x = np.mgrid[0:displacement_x.shape[0]:step,
#            0:displacement_x.shape[1]:step]
#     u = displacement_x[y, x]
#     v = displacement_y[y, x]
#
#     max_uv = max(np.abs(u).max(), np.abs(v).max())
#     if max_uv > 0:
#         u /= max_uv
#         v /= max_uv
#
#     plt.figure(figsize=(10, 10))
#     plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='r',
#                headwidth=3, headlength=5)
#     plt.title("Normalized Displacement Field")
#     plt.show()


def generate_random_corners(image_size, max_offset_percent=20):
    """改进角点扰动策略，防止过强畸变"""
    h, w = image_size
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    # 对称扰动增强稳定性
    offsets = np.random.uniform(-max_offset_percent / 100, max_offset_percent / 100,
                                (4, 2)) * [w, h]
    dst_pts = np.clip(src_pts + offsets, [0, 0], [w - 1, h - 1])
    return dst_pts.astype(np.float32)


def fixed_size_perspective_transform(image, src_pts, dst_pts):
    """核心函数：确保输出尺寸不变，并精准计算位移场"""
    h, w = image.shape[:2]

    # 扩展图像尺寸以保留所有信息
    expand_ratio = 1.5
    expanded_image = cv2.copyMakeBorder(image, int(h * expand_ratio), int(h * expand_ratio),
                                        int(w * expand_ratio), int(w * expand_ratio),
                                        cv2.BORDER_REFLECT_101)

    # 调整角点坐标至扩展后图像
    offset_x = int(w * expand_ratio)
    offset_y = int(h * expand_ratio)
    exp_src_pts = src_pts + [offset_x, offset_y]
    exp_dst_pts = dst_pts + [offset_x, offset_y]
    # 确保数据类型为 np.float32
    exp_src_pts = exp_src_pts.astype(np.float32)
    exp_dst_pts = exp_dst_pts.astype(np.float32)
    # 计算扩展图像的透视变换矩阵
    M = cv2.getPerspectiveTransform(exp_src_pts, exp_dst_pts)
    transformed = cv2.warpPerspective(expanded_image, M, (int(w * (1 + 2 * expand_ratio)),
                                                          int(h * (1 + 2 * expand_ratio))),
                                      # borderMode=cv2.BORDER_REFLECT_101
                                      )

    # 计算逆向坐标映射关系
    M_inv = cv2.getPerspectiveTransform(exp_dst_pts, exp_src_pts)

    # 生成扩展后图像的坐标网格
    y_grid, x_grid = np.mgrid[offset_y:offset_y + h, offset_x:offset_x + w]
    coords = np.stack([x_grid.ravel(), y_grid.ravel(), np.ones_like(x_grid.ravel())])

    # 逆向映射至原扩展图像坐标
    mapped = M_inv @ coords
    mapped = (mapped[:2] / mapped[2]).reshape(2, h, w)
    disp_x = mapped[0] - x_grid  # 目标坐标到原图坐标的位移
    disp_y = mapped[1] - y_grid

    # 裁切回原图区域
    final_image = transformed[offset_y:offset_y + h, offset_x:offset_x + w]

    return final_image, disp_x.astype(np.float32), disp_y.astype(np.float32)


def correct_distortion(distorted_image, disp_x, disp_y):
    """高精度校正畸变"""
    h, w = distorted_image.shape[:2]

    # 生成网格坐标
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    # 确保 disp_x 和 disp_y 的形状与 x_grid 一致
    if disp_x.shape != x_grid.shape:
        disp_x = disp_x.transpose(1, 0)
        disp_y = disp_y.transpose(1, 0)

    # 计算反向映射坐标
    map_x = (x_grid - disp_x).astype(np.float32)
    map_y = (y_grid - disp_y).astype(np.float32)

    # 使用 remap 进行反向映射
    corrected_image = cv2.remap(
        distorted_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        # borderMode=cv2.BORDER_REFLECT_101
    )

    return corrected_image


# 主程序测试
if __name__ == "__main__":
    image = cv2.imread("E:\photo\chess board.jpg")
    image = cv2.resize(image,[512,512])
    # image = cv2.cvtColor(cv2.imread("E:\photo\dog.png"), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # 生成带位移场的固定尺寸透视变换
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_pts = generate_random_corners((h, w), max_offset_percent=20)
    transformed, disp_x, disp_y = fixed_size_perspective_transform(image, src_pts, dst_pts)

    # 应用校正
    corrected = correct_distortion(transformed, disp_x, disp_y)
    cv2.imshow('ori',image)
    cv2.imshow('distortion_img',transformed)
    cv2.imshow('correct',corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # 可视化结果
    # plt.figure(figsize=(15, 5))
    # plt.subplot(131), plt.title("Original"), plt.imshow(image)
    # plt.subplot(132), plt.title("Distorted"), plt.imshow(transformed)
    # plt.subplot(133), plt.title("Corrected"), plt.imshow(corrected)
    # plt.tight_layout()
    # plt.show()
    # print(disp_x,disp_y)
    # 位移场可视化
    visualize_displacement_field(disp_x, disp_y)