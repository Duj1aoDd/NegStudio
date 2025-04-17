import cv2
import numpy as np
import math

class NEG:
    def __init__(self, path):
        self.attempt = 1
        self.img = cv2.imread(path)
        self.img_broaderLinesDebug = self.img.copy()
        self.corners = []
        self.Hbroader = []
        self.Vbroader = []
        self.Hlines = []
        self.Vlines = []
        self.Hlines_averageY = []
        self.Vlines_averageX = []
        self.HV = []
        self.angle = []
        self.tangent = []
        self.linesXY = []
        self.lines = []
        self.edges = []
        self.gray = []
        if self.img is None:
            raise ValueError("无法读取图像，请检查路径")
        else:
            print("图像读取成功", self.img.shape)
        self.upperBroad = 0
        self.lowerBroad = 0
        self.rightBroad = 0
        self.leftBroad = 0
    def blackBroadVerify(self):
        if len(self.corners) != 4:
            return False  # 如果角点检测不正确，返回 False
        rd,ld,lu,ru = self.corners
        # 将坐标转换为整数
        U = [int((lu[0] + ru[0]) / 2), int((lu[1] + ru[1]) / 2)]
        R = [int((ru[0] + rd[0]) / 2), int((ru[1] + rd[1]) / 2)]
        D = [int((ld[0] + rd[0]) / 2), int((ld[1] + rd[1]) / 2)]
        L = [int((lu[0] + ld[0]) / 2), int((lu[1] + ld[1]) / 2)]
        
        try:
            # 检查属性是否存在
            if not hasattr(self, 'blackField'):
                self.blackField =60  # 越高切得越狠
            # 检查是否有 gray 属性
            if not hasattr(self, 'gray') or self.gray_for_blackfieldTeest is None or self.gray_for_blackfieldTeest.size == 0:
                print("gray 属性未正确初始化，请检查。")
                return False
            if self.gray_for_blackfieldTeest[U[1], U[0]] < self.blackField:
                self.upperBroad = self.upperBroad+1
            if self.gray_for_blackfieldTeest[R[1], R[0]] < self.blackField:
                self.rightBroad = self.rightBroad+1
            if self.gray_for_blackfieldTeest[D[1], D[0]] < self.blackField:
                self.lowerBroad = self.lowerBroad+1
            if self.gray_for_blackfieldTeest[L[1], L[0]] < self.blackField:
                self.leftBroad = self.leftBroad+1

            return self.gray_for_blackfieldTeest[U[1], U[0]] < self.blackField or self.gray_for_blackfieldTeest[R[1], R[0]] < self.blackField or self.gray_for_blackfieldTeest[D[1], D[0]] < self.blackField or self.gray_for_blackfieldTeest[L[1], L[0]] < self.blackField
        except IndexError:
            print("blackBroadVerify 中出现索引越界错误，请检查角点坐标。")
            return False
    def sobel_edge_detection(self, img):
        #img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3) 
        sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3) 
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
        return gradient_magnitude 
    def cutBlack(self,img):
        Vsize = img.shape[0]
        Hsize = img.shape[1]
        for i in range(Vsize):
            for j in range(Hsize):
                if img[i][j] > 50:
                    img[i][j] = 0
        return img

    def broaderlinesDebug(self):
        if self.Hbroader is not None:
            for line in self.Hbroader:
                # 确保每个line是包含四个坐标的数组
                if isinstance(line, np.ndarray) and line.shape == (4,):  # 确保line是一个四个元素的数组
                    x1, y1, x2, y2 = line
                    cv2.line(self.img_broaderLinesDebug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                else:
                    print("无效的直线数据：", line)
        if self.Vbroader is not None:
            for line in self.Vbroader:
                # 确保每个line是包含四个坐标的数组
                if isinstance(line, np.ndarray) and line.shape == (4,):
                    x1, y1, x2, y2 = line
                    cv2.line(self.img_broaderLinesDebug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                else:
                    print("无效的直线数据：", line)
        cv2.imshow('Broader Lines Debug', self.img_broaderLinesDebug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def broaderDebug(self):
        # 重新初始化调试图像
        self.img_broaderLinesDebug = self.img.copy()

        # 只绘制通过corners得到的四条连接线段
        if self.corners:
            if len(self.corners) == 4:
                # 顺时针顺序连接四个点
                for i in range(4):
                    x1, y1 = self.corners[i]
                    x2, y2 = self.corners[(i + 1) % 4]  # 环绕连接
                    cv2.line(self.img_broaderLinesDebug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)  # 绿色

        # 显示图像
        cv2.imshow('Broader Debug', self.img_broaderLinesDebug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def HoughTF(self):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gray_for_blackfieldTeest = self.gray.copy()
        self.gray = self.sobel_edge_detection(self.gray)
        self.gray = self.cutBlack(self.gray)
        self.edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)
        self.lines = cv2.HoughLinesP(self.edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=50)
        if self.lines is None:
            raise ValueError("没有检测到任何直线")
        self.linesXY = self.lines[:, 0, :]
        return self.linesXY
    
    def __findTangent(self):
        self.tangent = []
        for line in self.linesXY:
            x1, y1, x2, y2 = line
            if x2 - x1 != 0:
                k = (y2 - y1) / (x2 - x1)
                self.tangent.append(k)
            else:
                self.tangent.append("V")
        return self.tangent
    
    def __tangent2angle(self):
        self.angle = []
        for k in self.tangent:
            if k == "V":
                self.angle.append(90)
                continue
            self.angle.append(math.atan(k) * 180 / math.pi)
        return self.angle
    
    def __findHV(self):
        self.HV = []
        for angle in self.angle:
            if abs(angle - 90) < 0.5 or abs(angle + 90) < 0.5:
                self.HV.append(1)#1表示垂直，-1表示水平
            elif abs(angle) < 0.5:
                self.HV.append(-1)
            else:
                self.HV.append(0)
        return self.HV
    
    def __findHVbroader(self):
        self.Hbroader = []
        self.Vbroader = []
        self.Hlines = []
        self.Vlines = []
        
        for i in range(len(self.HV)):
            if self.HV[i] == 1:
                self.Vlines.append(self.linesXY[i])
            elif self.HV[i] == -1:
                self.Hlines.append(self.linesXY[i])
            
        
        self.Hlines_averageY = [(line[1] + line[3]) / 2 for line in self.Hlines]
        self.Vlines_averageX = [(line[0] + line[2]) / 2 for line in self.Vlines]

        self.UHlines_averageY = self.Hlines_averageY.copy()
        self.LHlines_averageY = self.Hlines_averageY.copy()
        self.RVlines_averageX = self.Vlines_averageX.copy()
        self.LVlines_averageX = self.Vlines_averageX.copy()

        self.LHlines_averageY.sort(reverse=True)
        self.UHlines_averageY.sort()
        self.RVlines_averageX.sort(reverse=True)
        self.LVlines_averageX.sort()

        index_Hbroader = [self.Hlines_averageY.index(self.LHlines_averageY[self.lowerBroad]), self.Hlines_averageY.index(self.UHlines_averageY[self.upperBroad])]
        index_Vbroader = [self.Vlines_averageX.index(self.RVlines_averageX[self.rightBroad]), self.Vlines_averageX.index(self.LVlines_averageX[self.leftBroad])]
        
        self.Hbroader = [self.Hlines[index_Hbroader[0]], self.Hlines[index_Hbroader[1]]]#self.hbroader的格式为[[x1, y1, x2, y2], [x1, y1, x2, y2]]且第一条直线为下边界，第二条直线为上边界
        self.Vbroader = [self.Vlines[index_Vbroader[0]], self.Vlines[index_Vbroader[1]]]#self.vbroader的格式为[[x1, y1, x2, y2], [x1, y1, x2, y2]]且第一条直线为右边边界，第二条直线为左边边界
        return self.Hbroader, self.Vbroader

    def line_to_polar(self, line):
        x1, y1, x2, y2 = line
        # 计算斜率
        if x2 - x1 != 0:
            m = (y2 - y1) / (x2 - x1)
            theta = math.atan(m)  # 计算直线的角度
        else:
            theta = math.pi / 2  # 垂直线的角度为 90 度（π/2）
            
        r = x1 * math.cos(theta) + y1 * math.sin(theta)

        return r, theta

    def polar_to_cartesian(self, r, theta):
        x1 = r * math.cos(theta)
        y1 = r * math.sin(theta)
        # 根据直线的角度生成第二个点
        x2 = x1 + 1000 * math.cos(theta + math.pi / 2)  # 生成第二个点
        y2 = y1 + 1000 * math.sin(theta + math.pi / 2)
        
        return [x1, y1, x2, y2]

    def solve_intersection(self, A1, B1, C1, A2, B2, C2):

        # 计算行列式 D = A1 * B2 - A2 * B1
        D = A1 * B2 - A2 * B1
        if D == 0:
            print("两条直线平行或重合，无交点")
            return None  # 直线平行或重合，没有交点

        # 计算交点的坐标
        Dx = C1 * B2 - C2 * B1
        Dy = A1 * C2 - A2 * C1
        x = Dx / D
        y = Dy / D
        return np.array([x, y])

    def line_to_general(self, line):
        x1, y1, x2, y2 = line
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return A, B, C

    def __findCorner(self):
        print("Hbroader:", self.Hbroader)
        print("Vbroader:", self.Vbroader)

        if len(self.Hbroader) < 2 or len(self.Vbroader) < 2:
            print("水平线或垂直线不足，无法计算交点")
            return

        # 获取水平线和垂直线的坐标
        H1 = self.Hbroader[0]
        H2 = self.Hbroader[1]
        V1 = self.Vbroader[0]
        V2 = self.Vbroader[1]

        # 获取水平线的参数 (A, B, C)
        A1, B1, C1 = self.line_to_general(H1)
        A2, B2, C2 = self.line_to_general(H2)

        # 获取垂直线的参数 (A, B, C)
        A3, B3, C3 = self.line_to_general(V1)
        A4, B4, C4 = self.line_to_general(V2)

        corner1 = self.solve_intersection(A1, B1, C1, A3, B3, C3)  # 左上
        corner2 = self.solve_intersection(A1, B1, C1, A4, B4, C4)  # 右上
        corner3 = self.solve_intersection(A2, B2, C2, A4, B4, C4)  # 右下
        corner4 = self.solve_intersection(A2, B2, C2, A3, B3, C3)  # 左下

        if corner1 is not None and corner2 is not None and corner3 is not None and corner4 is not None:
            self.corners = [-1*corner1, -1*corner2, -1*corner3, -1*corner4]
        else:
            print("某些交点计算失败")
            self.corners = []

        print("corners:", self.corners)       
        
    def FindBroaders(self):
        self.HoughTF()
        self.__findTangent()
        self.__tangent2angle()
        self.__findHV()
        self.__findHVbroader()
        self.__findCorner()
        print("corners:",self.corners)
        while self.blackBroadVerify():
            print("attempt:",self.attempt,"upbroad:",self.upperBroad,"lowbroad:",self.lowerBroad,"rightbroad:",self.rightBroad,"leftbroad:",self.leftBroad)
            self.HoughTF()
            self.__findTangent()
            self.__tangent2angle()
            self.__findHV()
            self.__findHVbroader()
            self.__findCorner()
            self.attempt = self.attempt + 1
        return self.corners
    
    def BGR2LAB(self,img):
        self.INVERTEDImg = 255-img
        self.LABImg = cv2.cvtColor(self.INVERTEDImg, cv2.COLOR_BGR2LAB)
        L,A,B = cv2.split(self.LABImg)
        return L,A,B
    
    def convertNEG(self):
        L,A,B = self.BGR2LAB(self.img)
        L= np.clip(L,0,255).astype(np.uint8)
        A = A +2
        B = B+26
        A = np.clip(A,0,255).astype(np.uint8)
        B = np.clip(B,0,255).astype(np.uint8)
        self.LABImg = cv2.merge((L,A,B))
        self.OUTPUT = cv2.cvtColor(self.LABImg, cv2.COLOR_LAB2BGR)
