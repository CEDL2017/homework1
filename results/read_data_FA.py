# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:10:49 2017

@author: nvlab
"""


import numpy as np
from PIL import Image
global k
k=0


def random_order(num, batch):
    order = np.random.randint(0, num-101,(1,batch))
    return order
    
    
def read_data(batch):
    png = []
    order = random_order(14992, batch)
    for i in range(batch):
        img_path = "frames_resize/Image%s.png"%order[0,i]
        
        img = Image.open(img_path)
        img = np.array(img)
        x_img = np.reshape(img, [1,95232])
        png.append(x_img)
                      
    png = np.array(png)
    png = np.reshape(png, [batch,95232])
    
    order_FA_label = np.zeros([batch, 2])
    
    for i in range(batch):
        order_FA_label[i,:] = FA_label[order[0,i],:]
        
    return png, order_FA_label
  

def read_data_test(batch):

    png = []
    order = random_order(12776, batch)
    for i in range(batch):
        img_path = "test_resize/Image%s.png"%order[0,i]
        
        img = Image.open(img_path)
        img = np.array(img)
        x_img = np.reshape(img, [1,95232])
        png.append(x_img)
                      
    png = np.array(png)
    png = np.reshape(png, [batch,95232])
    
    order_FA_label = np.zeros([batch, 2])
    
    for i in range(batch):
        order_FA_label[i,:] = FA_label[order[0,i],:]  
        
    return png, order_FA_label
    
def read_data_test_(batch):
    png = []
    global k
    for i in range(batch):
        img_path = "test_resize/Image%s.png"%k
        img = Image.open(img_path)
        img = np.array(img)
        x_img = np.reshape(img, [1,95232])
        png.append(x_img)
        k+=1
        
    png = np.array(png)
    png = np.reshape(png, [batch,95232])
              
    return png
    
def read_FA_labels():

    obj_label = []
    num_obj_left_label = np.zeros([14992,2])

    k = 1
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_left%s.npy"%k)
    obj_label = np.array(obj_left_labels)    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
        
    k =2
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =3
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =1
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =2
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =3
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =4
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =1
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =2
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =3
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    obj_label = np.array(obj_label)
    obj_label = np.reshape(obj_label,[1,14992]) 
    for i in range(0,14992):
        a = obj_label[0,i]
        #print(a)
        num_obj_left_label[i,int(a)] = 1

    return num_obj_left_label 
    
def read_FA_labels_test():

    obj_label = []
    num_obj_left_label = np.zeros([12776,2])

    k = 4
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_left%s.npy"%k)
    obj_label = np.array(obj_left_labels)    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
        
    k =5
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =6
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =5
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =6
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =7
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =8
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =4
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =5
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =6
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/FA_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    obj_label = np.array(obj_label)
    obj_label = np.reshape(obj_label,[1,12776]) 
    for i in range(0,12776):
        a = obj_label[0,i]
        #print(a)
        num_obj_left_label[i,int(a)] = 1
        
    ges_test_label = np.zeros([12736, 2])
    
    for i in range(0,12736):
        ges_test_label[i,:] = num_obj_left_label[i,:]

    return np.array(ges_test_label.astype(int)) 
    
FA_label = read_FA_labels()

FA_test_label = read_FA_labels_test()
