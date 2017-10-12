# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:10:49 2017

@author: nvlab
"""

import numpy as np
from PIL import Image


def random_order(num, batch):
    order = np.random.randint(0, num-1,(1,batch))
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
    
    order_obj_label = np.zeros([batch, 24])
    
    for i in range(batch):
        order_obj_label[i, :] = obj_label[order[0,i],:]
        
        
    return png, order_obj_label
    

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
    
    order_obj_label = np.zeros([batch, 24])
    
    for i in range(batch):
        order_obj_label[i, :] = obj_label[order[0,i],:]   
        
    return png, order_obj_label

def read_obj_labels():

    obj_label = []
    num_obj_left_label = np.zeros([14992,24])

    k = 1
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_left%s.npy"%k)
    obj_label = np.array(obj_left_labels)    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
        
    k =2
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =3
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =1
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =2
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =3
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =4
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =1
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =2
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =3
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    obj_label = np.array(obj_label)
    obj_label = np.reshape(obj_label,[1,14992]) 
    for i in range(0,14992):
        a = obj_label[0,i]
        #print(a)
        num_obj_left_label[i,int(a)] = 1

    return num_obj_left_label   
    
def read_obj_labels_test():

    obj_label = []
    num_obj_left_label = np.zeros([12776,24])

    k = 4
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_left%s.npy"%k)
    obj_label = np.array(obj_left_labels)    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
        
    k =5
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =6
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/house/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =5
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =6
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =7
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =8
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/lab/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    k =4
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    k =5
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))

    k =6
    obj_left_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_left%s.npy"%k)
    obj_label = np.hstack((obj_label, np.array(obj_left_labels)))    
    obj_right_labels = np.load("../../../../Disk2/cedl/handcam/labels/office/obj_right%s.npy"%k)
    obj_label = np.hstack((obj_label,np.array(obj_right_labels)))
    
    obj_label = np.array(obj_label)
    obj_label = np.reshape(obj_label,[1,12776]) 
    for i in range(0,12776):
        a = obj_label[0,i]
        #print(a)
        num_obj_left_label[i,int(a)] = 1

    return num_obj_left_label   
    
obj_label = read_obj_labels()

obj_test_label = read_obj_labels_test()
