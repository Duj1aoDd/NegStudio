import cv2
import numpy as np

def sobel_edge_detection(img):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3) 
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    return gradient_magnitude
# 读取图像并转换为灰度
img = cv2.imread('/Users/yichen/Documents/NegStudio/testImg3.tiff')
if img is None:
    raise ValueError("无法读取图像，请检查路径")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = sobel_edge_detection(gray)

# 添加高斯模糊步
#blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# 边缘检测
edges = cv2.Canny(gray, 80, 150, apertureSize=3)

# 霍夫变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40 , maxLineGap=50)

# 绘制直线
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

# 显示结果
cv2.imshow('Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(lines[:,0,:])
