


from skimage import io,transform
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import glob
import os
import re
import numpy as np
import time
import tensorflow as tf

import matplotlib.pyplot as plt


#from PIL import Image
#####################################

reload_net=0              #是否Load   1是0否
reload_net_path='save_model1'        #reload的位置
savenet = 1               #是否保存        1是0否
save_net_path= 'save_model'         #本次保存位置



pathh1='./Data/test/house/1/Lhand/'      
pathh2='./Data/test/house/1/Rhand/'
pathh3='./Data/test/house/2/Lhand/'      
pathh4='./Data/test/house/2/Rhand/'
pathh5='./Data/test/house/3/Lhand/'      
pathh6='./Data/test/house/3/Rhand/'

pathl1='./Data/test/lab/1/Lhand/'      
pathl2='./Data/test/lab/1/Rhand/'
pathl3='./Data/test/lab/2/Lhand/'      
pathl4='./Data/test/lab/2/Rhand/'
pathl5='./Data/test/lab/3/Lhand/'      
pathl6='./Data/test/lab/3/Rhand/'
pathl7='./Data/test/lab/4/Lhand/'      
pathl8='./Data/test/lab/4/Rhand/'

patho1='./Data/test/office/1/Lhand/'      
patho2='./Data/test/office/1/Rhand/'
patho3='./Data/test/office/2/Lhand/'      
patho4='./Data/test/office/2/Rhand/'
patho5='./Data/test/office/3/Lhand/'      
patho6='./Data/test/office/3/Rhand/'


labelpath_house='./Data/labels/house/trainlabel/'      
labelpath_lab='./Data/labels/lab/trainlabel/'
labelpath_office='./Data/labels/office/trainlabel/'

testlabelpath_house='./Data/labels/house/testlabel/'      
testlabelpath_lab='./Data/labels/lab/testlabel/'
testlabelpath_office='./Data/labels/office/testlabel/'


model_dir = '/home/benita/Downloads/CEDLHW1/Alexnet/save_model'       
model_name = "ckp"
#參數設定
w=100             #size
h=100
c=3               #channel

n_epoch=1            #epoch
batch_size=20           #batchsize

################需要自己设定的部分#####################
class_names = ['free','computer','cellphone','coin','ruler','thermos-bottle','whiteboard-pen','whiteboard-eraser',
               'pen','cup','remote-control-TV','remote-control-AC','switch','windows','fridge','cupboard','water-tap',
               'toy','kettle','bottle','cookie','book','magnet','lamp-switch']





#讀取圖片路徑 house lab office
def read_test_list(): 

    test_path_list=[]
    test_path_listr=[]
    
    for im in os.listdir(pathh1):
        
        test_path = os.path.join(pathh1, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(pathh2):
        
        test_path = os.path.join(pathh2, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)
    test_path_list=[]   
    for im in os.listdir(pathh3):
        
        test_path = os.path.join(pathh3, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(pathh4):
        
        test_path = os.path.join(pathh4, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)
    test_path_list=[]    
    for im in os.listdir(pathh5):
        
        test_path = os.path.join(pathh5, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(pathh6):
        
        test_path = os.path.join(pathh6, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)
    test_path_list=[]



    for im in os.listdir(pathl1):
        
        test_path = os.path.join(pathl1, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(pathl2):
        
        test_path = os.path.join(pathl2, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)    
    test_path_list=[]   

    for im in os.listdir(pathl3):
        
        test_path = os.path.join(pathl3, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(pathl4):
        
        test_path = os.path.join(pathl4, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)    
    test_path_list=[]

    for im in os.listdir(pathl5):
        
        test_path = os.path.join(pathl5, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(pathl6):
        
        test_path = os.path.join(pathl6, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)    
    test_path_list=[]

    for im in os.listdir(pathl7):
        
        test_path = os.path.join(pathl7, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(pathl8):
        
        test_path = os.path.join(pathl8, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)    
    test_path_list=[]


    for im in os.listdir(patho1):
        
        test_path = os.path.join(patho1, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(patho2):
        
        test_path = os.path.join(patho2, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)    
    test_path_list=[]    

    for im in os.listdir(patho3):
        
        test_path = os.path.join(patho3, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(patho4):
        
        test_path = os.path.join(patho4, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)    
    test_path_list=[]

    for im in os.listdir(patho5):
        
        test_path = os.path.join(patho5, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

    test_path_listr.extend(test_path_list)
    test_path_list=[]
        
    for im in os.listdir(patho6):
        
        test_path = os.path.join(patho6, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))
    test_path_listr.extend(test_path_list)    
    test_path_list=[]    
    
    return np.asarray(test_path_listr) 
    
def read_img_list():  
    imgs_path_list=[]
    """
    for i in range(10):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/1/Lhand/Image'+str(i)+'.png'
        #im1 = Image.open(st)
        im = io.imread(st)
        #im = np.resize(im,(256,256,3))
        
        img.append(im)
        
    return np.asarray(img,np.float32)
    """
        
    """
    for i in range(831):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/1/Lhand/Image'+str(i)+'.png'
        #im1 = Image.open(st)
        im = io.imread(st)
        #im = np.resize(im,(256,256,3))
        
        img.append(im)
        
        #print(img)
        #im = np.asarray(im,np.float32)
    for i in range(831):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/1/Rhand/Image'+str(i)+'.png'
        #im1 = Image.open(st)
        im = io.imread(st)
        #im = np.resize(im,(256,256,3))
        
        img.append(im)   
        
    for i in range(988):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/2/Lhand/Image'+str(i)+'.png'
        #im1 = Image.open(st)
        im = io.imread(st)
        #im = np.resize(im,(256,256,3))
        
        img.append(im) 
    
    for i in range(988):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/2/Rhand/Image'+str(i)+'.png'
        #im1 = Image.open(st)
        im = io.imread(st)
        #im = np.resize(im,(256,256,3))
        
        img.append(im)
    
    for i in range(1229):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/3/Lhand/Image'+str(i)+'.png'
        #im1 = Image.open(st)
        im = io.imread(st)
        #im = np.resize(im,(256,256,3))
        
        img.append(im)

    for i in range(1229):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/3/Rhand/Image'+str(i)+'.png'
        #im1 = Image.open(st)
        im = io.imread(st)
        #im = np.resize(im,(256,256,3))
        
        img.append(im)
        
    print('house done')   




    for i in range(501):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/1/Lhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)
    for i in range(501):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/1/Rhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)

    for i in range(589):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/2/Lhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)

    for i in range(589):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/2/Rhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)   

    for i in range(730):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/3/Lhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)

    for i in range(730):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/3/Rhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)     

    for i in range(660):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/4/Lhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)

    for i in range(660):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/4/Rhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)


    print('lab done')


    for i in range(745):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/1/Lhand/Image'+str(i)+'.png'
        im = io.imread(st)
        iimg.append(im)

    for i in range(745):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/1/Rhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)

    for i in range(572):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/2/Lhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)

    for i in range(572):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/2/Rhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)

    for i in range(651):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/3/Lhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)

    for i in range(651):
        i=i+1
        st='/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/3/Rhand/Image'+str(i)+'.png'
        im = io.imread(st)
        img.append(im)    
    return np.asarray(img)
    """
    
    
    #for i in range()
    for i in range(831):
    #for im in os.listdir(pathh1):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/1/Lhand/Image'+str(i)+'.png'
        #img_path = os.path.join(pathh1, im)
        #img_path=natsorted(img_path, key=lambda y: y.lower())
        
        
        #img_path = os.path.join(pathh1, im)
        
        #img_path=img_path.sort(key = lambda x:int(x[:-4]))
        #img_path = sorted(img_path, key = lambda x: int(x.split("_")[1]))
        #print(img_path)
        #img=io.imread(os.path.join(path1, im))
        #img=transform.resize(img,(w,h,c))
        imgs_path_list.append(img_path)
        
    for i in range(831):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/1/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(988):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/2/Lhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(988):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/2/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(1229):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/3/Lhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(1229):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/3/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)




    for i in range(501):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/1/Lhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(501):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/1/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(589):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/2/Lhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(589):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/2/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(730):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/3/Lhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(730):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/3/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)


    for i in range(660):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/4/Lhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(660):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/lab/4/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)




    for i in range(745):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/1/Lhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(745):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/1/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(572):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/2/Lhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)


    for i in range(572):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/2/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(651):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/3/Lhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

    for i in range(651):
        i=i+1
        img_path = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/office/3/Rhand/Image'+str(i)+'.png'
        imgs_path_list.append(img_path)
       
    return np.asarray(imgs_path_list)



"""
for im in os.listdir(pathl1):
test_path = os.path.join(pathl1, im)
#img=io.imread(os.path.join(path1, im))
#img=transform.resize(img,(w,h,c))
imgs_path_list.append(test_path)
for im in os.listdir(pathl2):
test_path = os.path.join(pathl2, im)
#img=io.imread(os.path.join(path1, im))
#img=transform.resize(img,(w,h,c))
imgs_path_list.append(test_path)
for im in os.listdir(pathl3):
test_path = os.path.join(pathl3, im)
#img=io.imread(os.path.join(path1, im))
#img=transform.resize(img,(w,h,c))
imgs_path_list.append(test_path)
for im in os.listdir(pathl4):
test_path = os.path.join(pathl4, im)
#img=io.imread(os.path.join(path1, im))
#img=transform.resize(img,(w,h,c))
imgs_path_list.append(test_path)
for im in os.listdir(pathl5):
test_path = os.path.join(pathl5, im)
#img=io.imread(os.path.join(path1, im))
#img=transform.resize(img,(w,h,c))
imgs_path_list.append(test_path)
for im in os.listdir(pathl6):
test_path = os.path.join(pathl6, im)
#img=io.imread(os.path.join(path1, im))
#for i in range(14992)

imgs_path_list.append(test_path) 
for im in os.listdir(pathl7):
test_path = os.path.join(pathl7, im)
#img=io.imread(os.path.join(path1, im))
#img=transform.resize(img,(w,h,c))
imgs_path_list.append(test_path)
for im in os.listdir(pathl8):
test_path = os.path.join(pathl8, im)
imgs_path_list.append(test_path) 

for im in os.listdir(patho1):
test_path = os.path.join(patho1, im)
imgs_path_list.append(test_path)
#print (test_path)
for im in os.listdir(patho2):
test_path = os.path.join(patho2, im)
imgs_path_list.append(test_path)
for im in os.listdir(patho3):
test_path = os.path.join(patho3, im)
imgs_path_list.append(test_path)
for im in os.listdir(patho4):
test_path = os.path.join(patho4, im)
imgs_path_list.append(test_path)
for im in os.listdir(patho5):
test_path = os.path.join(patho5, im)
imgs_path_list.append(test_path)
for im in os.listdir(patho6):
test_path = os.path.join(patho6, im)
imgs_path_list.append(test_path)


    return np.asarray(img,np.float32)
    """

def read_labels():   
    #imgs=[]
    labelz=[]
    labels=[]
    labelss=[]
    lbls=[]
    #print(os.listdir(labelpath_house)
    for i in range(3):
        i=i+1
        pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/house/trainlabel/L/obj_left'+str(i)+'.npy'
        #pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/house/trainlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathl)
        labelz.append(labels)
    for i in range(3):
        i=i+1
        #pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/house/trainlabel/L/obj_left'+str(i)+'.npy'
        pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/house/trainlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathr)
        labelz.append(labels)       
        #print(labelz.shape)
        
    for i in range(4):
        i=i+1
        pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/L/obj_left'+str(i)+'.npy'
        #pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathl)
        labelz.append(labels)
    for i in range(4):
        i=i+1
        #pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/L/obj_left'+str(i)+'.npy'
        pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathr)
        labelz.append(labels)  


    for i in range(3):
        i=i+1
        pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/office/trainlabel/L/obj_left'+str(i)+'.npy'
        #pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathl)
        labelz.append(labels)
    for i in range(3):
        i=i+1
        #pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/L/obj_left'+str(i)+'.npy'
        pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/office/trainlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathr)
        labelz.append(labels)  
    return np.concatenate(labelz)

def read_test_labels():   

    labelz=[]
    labels=[]
    labelss=[]
    lbls=[]
    #print(os.listdir(labelpath_house)
    for i in range(3):
        i=i+4
        pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/house/testlabel/L/obj_left'+str(i)+'.npy'
        #pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/house/trainlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathl)
        labelz.append(labels)
    for i in range(3):
        i=i+4
        #pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/house/trainlabel/L/obj_left'+str(i)+'.npy'
        pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/house/testlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathr)
        labelz.append(labels)       
        #print(labelz.shape)
        
    for i in range(4):
        i=i+5
        pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/testlabel/L/obj_left'+str(i)+'.npy'
        #pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathl)
        labelz.append(labels)
    for i in range(4):
        i=i+5
        #pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/L/obj_left'+str(i)+'.npy'
        pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/testlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathr)
        labelz.append(labels)  


    for i in range(3):
        i=i+4
        pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/office/testlabel/L/obj_left'+str(i)+'.npy'
        #pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathl)
        labelz.append(labels)
    for i in range(3):
        i=i+4
        #pathl = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/lab/trainlabel/L/obj_left'+str(i)+'.npy'
        pathr = '/home/benita/Downloads/CEDLHW1/Alexnet/Data/labels/office/testlabel/R/obj_right'+str(i)+'.npy'
        labels = np.load(pathr)
        labelz.append(labels)  
    return np.concatenate(labelz)

data_list=read_img_list()
data_list1=read_test_list()
#print(data_list)
label=read_labels()
testlabel=read_test_labels()
#print(label.shape[0])



#打亂順序
num_example=data_list.shape[0]
#print(num_example)
arr=np.arange(num_example)
np.random.shuffle(arr)
data_list=data_list[arr]
#print(data_list)
label=label[arr]
#print(label.shape[0])

num=data_list1.shape[0]
#print(num)
arr=np.arange(num)
np.random.shuffle(arr)
data_list1=data_list1[arr]
#print(data_list)
testlabel=testlabel[arr]
#print(testlabel.shape[0])


"""
ratio=0.8
s=np.int(num_example*ratio)
x_train=data_list[:s]
#print(x_train)
y_train=label[:s]
x_val=data_list[s:]
y_val=label[s:]
"""

x_train=data_list
y_train=label
x_val=data_list1
y_val=testlabel


#-----------------建構網路----------------------

x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

#（100——>50) 
#200 - 100 
conv1=tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

#(50->25)
#100 - 50
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#(25->12)
#50 - 25
conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

#(12->6)
#25 - 12
conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
"""
#6 -> 3
#12 - 6
conv5=tf.layers.conv2d(
      inputs=pool4,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool5=tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

#3->1
#6 -3

conv6=tf.layers.conv2d(
      inputs=pool5,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool6=tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

conv7=tf.layers.conv2d(
      inputs=pool6,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool7=tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
"""
"""
conv8=tf.layers.conv2d(
      inputs=pool7,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool8=tf.layers.max_pooling2d(inputs=conv8, pool_size=[2, 2], strides=2)
"""

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

#FC
dense1 = tf.layers.dense(inputs=re1, 
                      units=512, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

dense2= tf.layers.dense(inputs=dense1, 
                      units=512, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

logits= tf.layers.dense(inputs=dense2, 
                        units=24, 
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
#---------------------------網路结束---------------------------
#print(logits.shape)
loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#批次取數據編號 
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
                
        yield inputs[excerpt], targets[excerpt]


#訓練


sess=tf.InteractiveSession()  
if reload_net==1:
    saver = tf.train.Saver()
    saver.restore(sess,model_dir+'/'+model_name)
    print('恢復數據成功')
else:
    sess.run(tf.global_variables_initializer())
    print ('初始化數據成功')
ep=0
for epoch in range(n_epoch):
    start_time = time.time()
    #training
    #print(io.imread('/home/benita/Downloads/CEDLHW1/Alexnet/Data/train/house/2/Rhand/Image542.png'))
    
    train_loss, train_acc, n_batch = 0, 0, 0 
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        #x_train_a = io.imread(x_train[])
        #print(tmp)
        #tmp = tmp**(1/1.5)  #gamma correction
        
        temp =[]
        #print(x_train_a)
        for i in range(batch_size):            
        #print(io.imread(x_train_a[1]))  
        
            tmp = io.imread(x_train_a[i])
            #print(i)
            tmp = np.resize(tmp,(w,h,c))
           # print(tmp)
            tmp = tmp/256
            #tmp = np.reshape(tmp,[-1,w,h,c])
            #print(tmp.shape)
            temp.append(tmp)
            #print(temp)
           #tmp = tmp/256
            
        #temp = np.reshape(temp,[batch_size,w,h,c])
        #print(temp.shape)    
        """
        tmp = io.imread(x_train_a[1])
        tmp = np.resize(tmp,(w,h,c))
        tmp = tmp/256
        tmp = np.reshape(tmp,[-1,w,h,c])
        temp = np.concatenate(tmp)        
        print(temp)
        temp = np.reshape(temp,[-1,w,h,c])
        """
        x_train_a = temp
              
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1

        
        
        #print(" precision  %f " % precision)
        if (n_batch%50 == 0):
            print("|  step     %d/15 | " % (n_batch/50))
            print("|  train acc:  %f | " % (train_acc/ n_batch))
            print("|  train loss: %f | " % (train_loss/ n_batch))
    print("   train loss: %f" % (train_loss/ n_batch))
    print("   train acc: %f" % (train_acc/ n_batch))
    
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        temp =[]
        for i in range(batch_size):            
        #print(io.imread(x_train_a[1]))  
        
            tmp = io.imread(x_val_a[i])
            tmp = np.resize(tmp,(w,h,c))
           # print(tmp)
            tmp = tmp/256
            #tmp = np.reshape(tmp,[-1,w,h,c])
            #print(tmp.shape)
            temp.append(tmp)        
        """
        #for j in enumerate(x_val_a):
        tmp = io.imread(x_val_a[0])
        #tmp = tmp**(1/1.5)
        tmp = np.resize(tmp,(w,h,c))
        tmp = tmp/256
        tmp = np.reshape(tmp,[-1,w,h,c])
        """    
        x_val_a = temp
	
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
        if (n_batch%50 == 0):
            print("|  step     %d/13 | " % (n_batch/50))
            print("|  validation acc:  %f | " % (val_acc/ n_batch))

    print("   validation loss: %f" % (val_loss/ n_batch))
    print("   validation acc: %f" % (val_acc/ n_batch))
    
    ep+=1
    print("epoch %d Finish" % ep)
val_loss, val_acc, n_batch = 0, 0, 0
for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
    temp =[]
    for i in range(batch_size):            
        #print(io.imread(x_train_a[1]))  
        
        tmp = io.imread(x_val_a[i])
        tmp = np.resize(tmp,(w,h,c))
           # print(tmp)
        tmp = tmp/256
            #tmp = np.reshape(tmp,[-1,w,h,c])
            #print(tmp.shape)
        temp.append(tmp)        

    """
    tmp = io.imread(x_val_a[0])
    tmp = np.resize(tmp,(w,h,c))
    tmp = tmp/256
    tmp = np.reshape(tmp,[-1,w,h,c])
     """       
    x_val_a = temp
    
    err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
    val_loss += err; val_acc += ac; n_batch += 1
print("  Final validation loss: %f" % (val_loss/ n_batch))
print("  Fianl validation acc: %f" % (val_acc/ n_batch))  
print("訓練完成！")

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
# 保存模型
saver = tf.train.Saver()
if (savenet==1):
    saver.save(sess, os.path.join(model_dir, model_name))
    print("保存模型成功！")
sess.close()
