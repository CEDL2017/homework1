# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:23:41 2017

@author: user
"""

import tensorflow as tf
import numpy as np
import os
import re

#%%

# you need to change this to your data directory
#train_dir = 'Lhand/'

def read_data(file_dir, label_dir, train = True):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    image_list = []
    image_temp = []
#    temp = []
    for root, direc, files in os.walk(file_dir):
        # image directories
        for name in files:
            image_temp.append(os.path.join(root, name))
        image_list = image_list + sorted(image_temp, key = lambda x: int(re.sub('\D', '', x)))
        image_temp = []
        # get 10 sub-folder names
#        for name in direc:
#            temp.append(os.path.join(root, name))
    
#    print(image_list)
#    temp = []  
    label_list = []  
    for root, direc, files in os.walk(label_dir):
#        print(root)
#        print(root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/lab')
        if train:
#            print("aaaaa")   
            if root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/lab':
#                print("111")
                for i in range(1, 5):
                    #print('temp = ', temp)
                    label_list.extend(np.load(os.path.join(root, 'obj_left'+ str(i) + '.npy')))
                    #print('label_list = ', label_list)
                for i in range(1, 5):
                    label_list.extend(np.load(os.path.join(root, 'obj_right'+ str(i) + '.npy')))
            if (root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/house') or (root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/office'):
#                print("222")
                for i in range(1, 4):
                    label_list.extend(np.load(os.path.join(root, 'obj_left'+ str(i) + '.npy')))
                for i in range(1, 4):
                    label_list.extend(np.load(os.path.join(root, 'obj_right'+ str(i) + '.npy')))
        else:
            if root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/lab':
                for i in range(5, 9):
                    label_list.extend(np.load(os.path.join(root, 'obj_left'+ str(i) + '.npy')))
                for i in range(5, 9):
                    label_list.extend(np.load(os.path.join(root, 'obj_right'+ str(i) + '.npy')))
            if(root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/house') or (root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/office'):
                for i in range(4, 7):
                    label_list.extend(np.load(os.path.join(root, 'obj_left'+ str(i) + '.npy')))
                for i in range(4, 7):
                    label_list.extend(np.load(os.path.join(root, 'obj_right'+ str(i) + '.npy')))

#    image_list = []
#    for file in os.listdir(file_dir):
#        image_list.append(file_dir + file)
#
#    image_list = sorted(image_list, key = lambda x: int(re.sub('\D', '', x)))
#    #print(image_list)
#    label_list = []
#    label_list = np.load(label_dir)
    
    #print(len(label_list))
#    print(label_list)
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    #label_list = tf.cast(label_list, tf.int32)
    label_list = [int(float(i)) for i in label_list]
    
    #print(label_list)
    
    return image_list, label_list


#%%

def batch_generate(image, label, image_W, image_H, batch_size, capacity):
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=3)
#    print(image)
#    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.resize_images(image, [image_W, image_H])
    image = tf.image.per_image_standardization(image)  
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    label_batch = tf.one_hot(label_batch, 24)
    label_batch = tf.cast(label_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size, 24])
    image_batch = tf.cast(image_batch, tf.float32)
    
    
#    label_batch = tf.reshape(label_batch, [batch_size])
#    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

