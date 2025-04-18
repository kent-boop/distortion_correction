import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def cv_show(name, img):
    cv.imshow(name, img)
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

# 畸变后滤波处理
def apply_filter(image, filter_type="gaussian", kernel_size=(5, 5)):
    if filter_type == "gaussian":
        return cv.GaussianBlur(image, kernel_size, 0)
    elif filter_type == "median":
        return cv.medianBlur(image, kernel_size[0])
    elif filter_type == "bilateral":
        return cv.bilateralFilter(image, 9, 75, 75)
    else:
        return image

# 归一化坐标
def normalize_coordinates(h, w):
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    return np.meshgrid(x, y)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
# 桶形畸变（三阶径向畸变）
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

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
# 枕形畸变（三阶径向畸变）
def pincushion_distortion_with_padding(image, k1, k2, k3, pad_ratio=0.5):
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
    r_distorted = r * (1 - k1 * r ** 2 - k2 * r ** 4 - k3 * r ** 6)  # 使用正系数

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

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#剪切畸变Shear Distortion
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

    # 计算位移场
    disp_x = np.zeros((dis_h, dis_w), dtype=np.float32)
    disp_y = np.zeros((dis_h, dis_w), dtype=np.float32)

    # 对输出图像的每个像素，计算其在原图中的坐标
    inv_M = cv.invertAffineTransform(M_total)  # 计算逆变换矩阵

    x_grid, y_grid = np.meshgrid(np.arange(dis_w), np.arange(dis_h))
    src_x = inv_M[0, 0] * x_grid + inv_M[0, 1] * y_grid + inv_M[0, 2]
    src_y = inv_M[1, 0] * x_grid + inv_M[1, 1] * y_grid + inv_M[1, 2]

    # 计算位移场
    disp_x =  (x_grid - src_x)
    disp_y =  (y_grid - src_y)

    return sheared_image, disp_x, disp_y

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#波浪畸变
def Combined_Wave_Distortion(A_h,A_v,lambda_h,lambda_v,image):
    h, w = image.shape[:2]
    pad_size = int(0.2 * max(h, w))
    image_padded = cv.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_REFLECT_101)
    h_pad, w_pad = image_padded.shape[:2]

    # 生成二维坐标网格（确保 map_x 和 map_y 尺寸一致）
    y_coords, x_coords = np.mgrid[0:h_pad, 0:w_pad]  # 关键修改：使用 np.mgrid 生成二维网格
    map_x = x_coords + A_h * np.sin(2 * np.pi * y_coords / lambda_h)
    map_y = y_coords + A_v * np.sin(2 * np.pi * x_coords / lambda_v)

    # 转换为 float32
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 应用畸变
    wave_image_padded = cv.remap(
        image_padded,
        map_x,
        map_y,
        interpolation=cv.INTER_LINEAR,
        # borderMode=cv.BORDER_REFLECT_101
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

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#透视畸变
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
    expanded_image = cv.copyMakeBorder(image, int(h * expand_ratio), int(h * expand_ratio),
                                        int(w * expand_ratio), int(w * expand_ratio),
                                        cv.BORDER_REFLECT_101)

    # 调整角点坐标至扩展后图像
    offset_x = int(w * expand_ratio)
    offset_y = int(h * expand_ratio)
    exp_src_pts = src_pts + [offset_x, offset_y]
    exp_dst_pts = dst_pts + [offset_x, offset_y]
    # 确保数据类型为 np.float32
    exp_src_pts = exp_src_pts.astype(np.float32)
    exp_dst_pts = exp_dst_pts.astype(np.float32)
    # 计算扩展图像的透视变换矩阵
    M = cv.getPerspectiveTransform(exp_src_pts, exp_dst_pts)
    transformed = cv.warpPerspective(expanded_image, M, (int(w * (1 + 2 * expand_ratio)),
                                                          int(h * (1 + 2 * expand_ratio))),
                                      # borderMode=cv2.BORDER_REFLECT_101
                                      )

    # 计算逆向坐标映射关系
    M_inv = cv.getPerspectiveTransform(exp_dst_pts, exp_src_pts)

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

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#旋转畸变
def rotate_image_with_displacement(image, angle):
    """旋转图像并保持原始尺寸"""
    h, w = image.shape[:2]
    pad_size = int(max(h, w) * 0.5) # 扩展50%边界

    # 扩展图像边界
    padded_img = cv.copyMakeBorder(image,pad_size,pad_size,pad_size,pad_size,cv.BORDER_REFLECT_101)

    # 计算新的中心点
    new_center = (w // 2 + pad_size, h // 2 + pad_size)

    # 创建旋转矩阵
    M = cv.getRotationMatrix2D(new_center, angle, 1.0)

    # 计算旋转后的完整图像
    rotated_padded = cv.warpAffine(padded_img, M,
                                    (w + 2 * pad_size, h + 2 * pad_size),
                                    flags=cv.INTER_CUBIC,
                                    # borderMode=cv2.BORDER_REFLECT_101
                                    )

    # 裁剪回原始尺寸
    rotated_image = rotated_padded[pad_size:-pad_size, pad_size:-pad_size]

    # 计算位移场
    inv_M = cv.invertAffineTransform(M)
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

# #调试各种畸变是否可以用
# #读取图像
# image = cv.imread("E:\photo\dog.png")
# image = cv.resize(image,[512,512])
# print(image.shape )
# #桶形畸变
# k1, k2, k3 = 0.15, 0.05, 0.01
# distortion_img,disp_x,disp_y = barrel_distortion_with_padding(image,k1,k2,k3)
# cv_show('barrel',distortion_img)
# print('size:',distortion_img.shape)
# visualize_displacement_field(disp_x,disp_y)
# #枕形畸变
# distortion_img,disp_x,disp_y = pincushion_distortion_with_padding(image,k1,k2,k3)
# cv_show('pincushion',distortion_img)
# print('size:',distortion_img.shape)
# visualize_displacement_field(disp_x,disp_y)
# #波浪畸变
# # 定义波浪参数
# A_h = 5  # 水平振幅
# A_v = 5  # 垂直振幅
# lambda_h = 200  # 水平波长
# lambda_v = 200  # 垂直波长
# distortion_img, disp_x_c, disp_y_c = Combined_Wave_Distortion(A_h, A_v, lambda_h, lambda_v, image)
# cv_show('wave',distortion_img)
# print('size:',distortion_img.shape)
# visualize_displacement_field(disp_x,disp_y)
# #剪切畸变
# distortion_img, disp_x, disp_y = Combined_Shear_with_displacement(image, 0.1, 0.1, 512, 512)
# cv_show('shear',distortion_img)
# print('size:',distortion_img.shape)
# visualize_displacement_field(disp_x,disp_y)
# #透视畸变
# h,w = image.shape[:2]
# # 生成带位移场的固定尺寸透视变换
# src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
# dst_pts = generate_random_corners((h, w), max_offset_percent=20)
# transformed, disp_x, disp_y = fixed_size_perspective_transform(image, src_pts, dst_pts)
# cv_show('perspective',distortion_img)
# print('size:',distortion_img.shape)
# visualize_displacement_field(disp_x,disp_y)
# #旋转畸变
# distortion_img, disp_x, disp_y = generate_random_rotation_with_displacement(image)
# print('size:',distortion_img.shape)
# # 显示结果
# cv.imshow("Rotated Image", distortion_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# # 可视化位移场
# visualize_displacement_field(disp_x, disp_y)
# 测试位移场一致性
def test_displacement_direction():
    # 创建测试图像(带网格或特征点)
    # img = np.zeros((512, 512, 3), dtype=np.uint8)
    img = cv.imread("E:\photo\chess board.jpg")
    cv.line(img, (256, 0), (256, 511), (0, 255, 0), 1)  # 垂直线
    cv.line(img, (0, 256), (511, 256), (0, 255, 0), 1)  # 水平线

    # 应用每种畸变并可视化位移场
    distortions = [
        ("Barrel", lambda: barrel_distortion_with_padding(img, 0.15, 0.05, 0.01)),
        ("Pincushion", lambda: pincushion_distortion_with_padding(img, 0.15, 0.05, 0.01)),
        ("Shear", lambda: Combined_Shear_with_displacement(img, 0.1, 0.1, 512, 512)),
        ("Wave", lambda: Combined_Wave_Distortion(5, 5, 200, 200, img)),
        ("Perspective", lambda: fixed_size_perspective_transform(img,
                                                                 np.float32([[0, 0], [511, 0], [511, 511], [0, 511]]),
                                                                 np.float32(
                                                                     [[50, 50], [461, 30], [481, 481], [30, 461]]))),
        ("Rotation", lambda: rotate_image_with_displacement(img, 15))
    ]

    for name, func in distortions:
        distorted, dx, dy = func()
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(cv.cvtColor(distorted, cv.COLOR_BGR2RGB))
        plt.title(f"{name} Distorted Image")

        plt.subplot(122)
        y, x = np.mgrid[0:512:20, 0:512:20]
        plt.quiver(x, y, dx[::20, ::20], dy[::20, ::20],
                   angles='xy', scale_units='xy', scale=1, color='r')
        plt.title(f"{name} Displacement Field")
        plt.show()


test_displacement_direction()
