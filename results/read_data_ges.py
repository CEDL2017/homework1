# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:10:49 2017

@author: nvlab
"""

import numpy as np
from PIL import Image


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
    
    order_ges_label = np.zeros([batch, 13])
    
    for i in range(batch):
        order_ges_label[i,:] = ges_label[order[0,i],:]
        
    return png, order_ges_label
    
def read_ges_labels():

    obj_label = []
    num_ges_label = np.zeros([14992,13])

    k = 1
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_left%s.npy"%k)
    obj_label = np.array(obj_left_labels)    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
        
    k =2
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =3
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =1
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =2
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =3
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =4
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =1
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =2
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =3
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    obj_label = np.array(obj_label)
    obj_label = np.reshape(obj_label,[1,14992]) 
    for i in range(0,14992):
        a = obj_label[0,i]
        #print(a)
        num_ges_label[i,int(a)] = 1

    return num_ges_label   

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
    
    order_ges_label = np.zeros([batch, 13])
    
    for i in range(batch):

        order_ges_label[i,:] = ges_label[order[0,i],:]    
        
    return png, order_ges_label
    
def read_ges_labels_test():

    obj_label = []
    num_ges_label = np.zeros([12776,13])

    k = 4
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_left%s.npy"%k)
    obj_label = np.array(obj_left_labels)    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
        
    k =5
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =6
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =5
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =6
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =7
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =8
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =4
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =5
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =6
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/ges_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    obj_label = np.array(obj_label)
    obj_label = np.reshape(obj_label,[1,12776]) 
    for i in range(0,12776):
        a = obj_label[0,i]
        #print(a)
        num_ges_label[i,int(a)] = 1

    return num_ges_label     
    
ges_label = read_ges_labels()

ges_test_label = read_ges_labels_test()
