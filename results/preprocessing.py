import numpy as np
from PIL import Image
import os

path0 = "frames"
testData = None
trainData = None
testLabel = None
trainLabel = None
size = 1920/24, 1080/24
lr = None
for train_test in os.listdir(path0):
    path1 = os.path.join(path0, train_test)
    for site in os.listdir(path1):
        path2 = os.path.join(path1, site)
        for num in os.listdir(path2):
            path3 = os.path.join(path2, num)
            for LR in os.listdir(path3):
                if LR == "head":
                    continue
                elif LR == "Lhand":
                    lr = "left"
                else:
                    lr = "right"
                path4 = os.path.join(path3, LR)
                print(path4)
                   
                for fileN in range(1,len(os.listdir(path4))+1):
                    file = os.path.join(path4,"Image"+str(fileN)+".png")
                    pngfile = Image.open(file)
                    pngfile.thumbnail(size, Image.ANTIALIAS)
                    temp = np.asarray( pngfile, dtype="uint8" )
                    if train_test == "train":
                        if trainData is None:
                            trainData = temp.reshape([1,-1])
                        else:
                            trainData = np.concatenate([trainData,temp.reshape([1,-1])])
                    else:
                        if testData is None:
                            testData = temp.reshape([1,-1])
                        else:
                            testData = np.concatenate([testData,temp.reshape([1,-1])])
                            
                if train_test == "train":
                    lfile = os.path.join("labels", site, "obj_"+lr+num+".npy")
                    temp = np.load(lfile).astype(np.int8)
                    if trainLabel is None:
                        trainLabel = temp[:len(os.listdir(path4))]
                    else: 
                        trainLabel = np.append(trainLabel, temp[:len(os.listdir(path4))])
                    print(trainData.shape, trainLabel.shape)
                else:
                    if site == "lab":
                        lfile = os.path.join("labels", site, "obj_"+lr+str(int(num)+4)+".npy")
                    else:
                        lfile = os.path.join("labels", site, "obj_"+lr+str(int(num)+3)+".npy")
                    temp = np.load(lfile).astype(np.int8)
                    if testLabel is None:
                        testLabel = temp[:len(os.listdir(path4))]
                    else: 
                        testLabel = np.append(testLabel, temp[:len(os.listdir(path4))])
                    print(testData.shape, testLabel.shape)
                        
b = np.zeros((testLabel.shape[0], 24))
b[np.arange(testLabel.shape[0]), testLabel] = 1
testLabel = b
b = np.zeros((trainLabel.shape[0], 24))
b[np.arange(trainLabel.shape[0]), trainLabel] = 1
trainLabel = b
print(testData.shape, trainData.shape, testLabel.shape, trainLabel.shape)
np.save("testData",testData)
np.save("trainData",trainData)
np.save("testLabel",testLabel)
np.save("trainLabel",trainLabel)