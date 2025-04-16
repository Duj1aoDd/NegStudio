from NEGstudio import NEG
Path = 'testImg.tiff'
neg = NEG(Path)
neg.FindBroaders()
neg.broaderlinesDebug()#边线检测结果
neg.broaderDebug()#边框检测结果
