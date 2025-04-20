from NEGstudio import NEG
import cv2
Path = 'testImg4.tiff'
neg = NEG(Path)
neg.FindBroaders()
neg.broaderlinesDebug()#边线检测结果
neg.broaderDebug()#边框检测结果
neg.sample()
print("LINERESULT:",)
neg.BGR2LAB(neg.img)
neg.convertNEG_v1()
neg.applyLUT()
cv2.imwrite('Converted.tiff', neg.output)
cv2.imshow('Converted Image', neg.output)
cv2.waitKey(0)
cv2.destroyAllWindows()