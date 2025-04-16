import cv2
import numpy as np
from NEGstudio import NEG
def sobel_edge_detection(image_path, output_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法读取图像")
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3) 
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    gradient_magnitude = NEG.cutBlack(None,gradient_magnitude)
    if output_path:
        cv2.imwrite(output_path, gradient_magnitude)
    return gradient_magnitude
if __name__ == "__main__":
    input_image = "testImg.tiff" 
    output_image = "edges.jpg"
    edges = sobel_edge_detection(input_image, output_image)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()