from NEGstudio import NEG
import cv2
Path = 'testImg.tiff'
neg = NEG(Path)
neg.FindBroaders()
neg.broaderlinesDebug()#边线检测结果
neg.broaderDebug()#边框检测结果
neg.convertNEG_v1()
cv2.imwrite('Converted.tiff', neg.converted)
cv2.imshow('Converted Image', neg.converted)
cv2.waitKey(0)
cv2.destroyAllWindows()