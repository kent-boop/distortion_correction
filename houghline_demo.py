import cv2 as cv
import numpy as np
#显示图像
def cv_show(name,img):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#霍夫变换检测直线
# 霍夫变换筛选函数
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

    print(f"Total length of detected lines: {total_length}")

    # 返回是否检测到的直线总长度
    return total_length>=lambda_*h

image = cv.imread(r"E:\photo\car_plate\000000.jpg")
image = cv.resize(image,[512,512])
cv_show('img',image)
a = has_sufficient_lines(image)
print(a)