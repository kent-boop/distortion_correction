from pathlib import Path
import random
import cv2 as cv
import numpy as np
import json
import apply_distortion
import time
random.seed(42)
np.random.seed(42)

#霍夫变换检测直线
def has_sufficient_lines(img, lambda_=4 , canny_threshold=100, hough_threshold=80):
    """检测图像中是否存在足够多的直线特征，并计算所有直线的总长度"""
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    # 边缘检测
    edges = cv.Canny(gray, canny_threshold // 2, canny_threshold)

    # 概率霍夫变换（检测线段而非无限长直线）
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,  # 决定线段检测响应阈值
        minLineLength=80,  # 线段最小长度
        maxLineGap=10  # 线段间最大间隙
    )

    # 如果未检测到直线，返回 False 和长度为 0
    if lines is None:
        return False, 0

    # 计算所有直线的总长度
    total_length = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]  # 提取直线端点
        length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)  # 计算直线长度
        total_length += length

    # print(f"Total length of detected lines: {total_length}")
    # 返回是否检测到的直线总长度
    return total_length>=lambda_*h


def batch_distortion(
        coco_dir,
        output_dir,
        dist_types,
        lambda_,
        skip_unqualified=True,
        mode="train",
        target_qualified_size=1000,
        max_attempts=5000
):
    # 合并训练集和验证集的图像路径
    train_dir = Path(coco_dir) / "train2017"/"train2017"
    val_dir = Path(coco_dir) / "val2017"/"val2017"
    all_img_paths = list(train_dir.glob("*.jpg")) + list(val_dir.glob("*.jpg"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    disp_field_dir = output_dir / "displacement_field"
    disp_field_dir.mkdir(parents=True, exist_ok=True)  # 自动创建子目录
    meta = []
    random.shuffle(all_img_paths)
    qualified_count = 0
    processed_count = 0
    start_time = time.time()
    # 初始化全局计数器
    global_counter = 0  # 用于生成000000格式的序号
    for img_path in all_img_paths:
        if qualified_count >= target_qualified_size :
            end_time = time.time()
            fin_time = end_time-start_time
            fin_time = fin_time/60
            print(f"畸变图像已到目标数量,所用时间为{fin_time}")
            break
        # elif processed_count >= max_attempts:
        #     print("超出输入原图最大范围")
        #     break
        processed_count += 1
        img = cv.imread(str(img_path))
        if img is None:
            continue
        img = cv.resize(img, [512, 512])
        if skip_unqualified and not has_sufficient_lines(img, lambda_=lambda_):
            continue

        h, w = img.shape[:2]
        qualified_count += 1
        # 遍历所有畸变类型，每种生成3张
        for dist_type in dist_types:
            for repeat in range(2):
                try:
                    # 为当前原图生成唯一基础名（6位数字 + 畸变类型）
                    name = f"{global_counter:06d}_{dist_type}"  # 格式化为000000, 000001等
                    global_counter +=1
                    # 原有畸变参数生成代码不变
                    if dist_type == "barrel":
                        k1 = random.choice([0, 0.15, 0.3])
                        k2 = random.choice([0, 0.05, 0.1])
                        k3 = random.choice([0, 0.025, 0.05])
                        distortion_img, disp_x, disp_y = apply_distortion.barrel_distortion_with_padding(img, k1,
                                                                                                         k2, k3)
                        params = {"type": "barrel", "k1": k1, "k2": k2, "k3": k3}

                    elif dist_type == "pincushion":
                        k1 = random.choice([0, 0.1, 0.2])
                        k2 = random.choice([0, 0.04, 0.08])
                        k3 = random.choice([0, 0.025, 0.05])
                        distortion_img, disp_x, disp_y = apply_distortion.pincushion_distortion_with_padding(img,
                                                                                                             k1, k2,
                                                                                                             k3)
                        params = {"type": "pincushion", "k1": k1, "k2": k2, "k3": k3}

                    elif dist_type == "wave":
                        A_h = random.choice([0, 3.5, 7])
                        A_v = random.choice([0, 3.5, 7])
                        lambda_h = random.choice([300, 7000, 1000])
                        lambda_v = random.choice([300, 7000, 1000])
                        distortion_img, disp_x, disp_y = apply_distortion.Combined_Wave_Distortion(A_h, A_v,
                                                                                                   lambda_h,
                                                                                                   lambda_v, img)
                        params = {"type": "wave", "A_h": A_h, "A_v": A_v, "lambda_h": lambda_h,
                                  "lambda_v": lambda_v}

                    elif dist_type == "rotate":
                        angle = random.choice([-30, 0, 30])  # angle>0:逆时针旋转，angle<0：顺时针旋转
                        distortion_img, disp_x, disp_y = apply_distortion.rotate_image_with_displacement(img, angle)
                        params = {"type": "rotate", "angle": angle}


                    elif dist_type == "perspective":
                        src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                        dst_pts = apply_distortion.generate_random_corners((h, w), max_offset_percent=20)
                        distortion_img, disp_x, disp_y = apply_distortion.fixed_size_perspective_transform(img, src_pts,
                                                                                                           dst_pts)
                        params = {"type": "perspective", "corners": dst_pts.tolist()}  # 转换为列表

                    elif dist_type == "shear":
                        alpha = random.choice([-0.08, 0, 0.08])
                        beta = random.choice([-0.08, 0, 0.08])
                        distortion_img, disp_x, disp_y = apply_distortion.Combined_Shear_with_displacement(img, alpha,
                                                                                                           beta, w, h)
                        params = {"type": "shear", "alpha": alpha, "beta": beta}

                    if global_counter%1000 ==0:
                        middle_time = time.time()
                        elapsed_time = middle_time - start_time
                        elapsed_time = elapsed_time/60
                        print(f"生成图像数目已达{global_counter},目前花费时间为{elapsed_time}")
                    # 保存畸变图像和位移场
                    dist_img_path = output_dir / f"{name}.jpg"
                    cv.imwrite(str(dist_img_path), distortion_img)
                    disp_x_path = disp_field_dir/ f"{name}_disp_x.npy"
                    disp_y_path = disp_field_dir/ f"{name}_disp_y.npy"
                    np.save(disp_x_path, disp_x)
                    np.save(disp_y_path, disp_y)

                    # 记录元数据
                    meta.append({
                        "original_id": img_path.stem,
                        "distorted_path": str(dist_img_path),
                        "disp_x_path": str(disp_x_path),
                        "disp_y_path": str(disp_y_path),
                        "params": params,
                        "distortion_type": dist_type,
                        "repeat_id": repeat
                    })


                except Exception as e:
                    print(f"Error processing {img_path.name} ({dist_type}-{repeat}): {str(e)}")
                    import traceback
                    traceback.print_exc()  # 打印完整堆栈信息
                    continue

    # 保存元数据
    with open(output_dir / "distortion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Completed. Qualified images: {qualified_count} (lambda_={lambda_})")

#主程序
output_dir = Path(r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\test")
output_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
output_dir = Path(r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\val")
output_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
output_dir = Path(r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\train")
output_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
#生成训练集（60000张）
batch_distortion(
    coco_dir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\data",
    output_dir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\train",
    dist_types=["barrel", "pincushion", "wave", "rotate", "perspective", "shear"],
    lambda_=4,
    skip_unqualified=True,
    mode="test",
    target_qualified_size=5000,#5000*12=60000
    max_attempts=50000
)
# 生成测试集（6000张）
batch_distortion(
    coco_dir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\data",
    output_dir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\test",
    dist_types=["barrel", "pincushion", "wave", "rotate", "perspective", "shear"],
    lambda_=4,
    skip_unqualified=True,
    mode="test",
    target_qualified_size=500,#500*12=6000
    max_attempts=50000
)
# 生成验证集（2400张）
batch_distortion(
    coco_dir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\data",
    output_dir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\val",
    dist_types=['barrel','pincushion','shear','rotate','perspective','wave'],
    #['barrel','pincushion','shear','rotate','perspective','wave']
    lambda_=0,
    skip_unqualified=True,
    mode="val",
    target_qualified_size=200,#200*12=2400
    max_attempts=50000
)