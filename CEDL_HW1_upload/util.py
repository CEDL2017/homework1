# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:29:29 2017

@author: PC-Tong
"""

import os
import tensorflow as tf
import numpy as np
from skimage import io
import cv2


DATA_PATH = "./Pre_process_data"
TFRECORD_NAME = "pre_process_data.tfrecords"
TFRECORD_TEST_NAME = "pre_process_test_data.tfrecords"

IMAGE_WIDTH = 192
IMAGE_HEIGHT = 108
CLASS_SIZE = 24
BATCH_SIZE = 16
EPOCH = 10

# path is a list of all data folders
def pre_process(is_train):
    
    if is_train:
        tfrecorder_filename = TFRECORD_NAME
    else:
        tfrecorder_filename = TFRECORD_TEST_NAME
    # list all path
    image_path, label_path = produce_path(is_train)
    
    # list all image and label names
    image_filename_list = []
    for dir_path in image_path:
        file_list = os.listdir(dir_path)
        for tmp in file_list:
            image_filename_list = np.append(image_filename_list, dir_path + tmp)
        
    label_list = []
    for dir_path in label_path:
        tmp = np.load(dir_path)
        label_list = np.append(label_list, tmp)
    #one_hot_label_list = tf.one_hot(indices = label_list, depth = CLASS_SIZE)
        
    # write to TFRecords
    writer = tf.python_io.TFRecordWriter(tfrecorder_filename)
    
    for image_filename, label in zip(image_filename_list, label_list):
        image = cv2.imread(image_filename)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation = cv2.INTER_CUBIC)
        #image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
        #image /= 256
        height, width, depth = image.shape
        image_string = image.tostring()
        #label_string = label.tostring()
        
        example = tf.train.Example(features = tf.train.Features(feature = {
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'image_string': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
                }))
        
        writer.write(example.SerializeToString())
        
    writer.close()
    
    
    
    
def load_file(is_train):
    if is_train:
        record_filename = TFRECORD_NAME
        filename_queue= tf.train.string_input_producer(
            [record_filename], num_epochs = EPOCH)
    else:
        record_filename = TFRECORD_TEST_NAME
        filename_queue= tf.train.string_input_producer(
            [record_filename], num_epochs = 1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    #print(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features = {
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'image_string': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.float32)
            })
    
    image = tf.decode_raw(features['image_string'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image = tf.reshape(image, [height, width, 3])
    
    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                    target_height=IMAGE_HEIGHT,
                                                    target_width=IMAGE_WIDTH)

    images, labels = tf.train.shuffle_batch(
            [resized_image, label],
            batch_size = BATCH_SIZE,
            capacity = 320,
            num_threads = 16,
            min_after_dequeue = 1
            )
    
    images = tf.cast(images, tf.float32)
    ## ONE-HOT      
    labels = tf.one_hot(labels, depth= CLASS_SIZE)
    labels = tf.cast(labels, dtype=tf.float32)
    labels = tf.reshape(labels, [BATCH_SIZE, CLASS_SIZE])
    
    return images, labels
    
    
    

def produce_path(is_train):
    position = ["house", "lab", "office"]
    count = [3, 4, 3]
    img_post_path = ["Lhand/", "Rhand/"]
    label_post_path = ["left", "right"]
    img_fore_path = "/home/viplab/Downloads/frames/"
    label_fore_path = "/home/viplab/Downloads/labels/"
    
    img = []
    label = []
    for i in range(len(position)):
        for j in range(1, count[i]+1):
            for k in range(2):
                if is_train:
                    img += [img_fore_path + "train/" + position[i] + "/" + str(j) + "/" + img_post_path[k]]
                    label += [label_fore_path + position[i] + "/" + "obj_" + label_post_path[k] + str(j) + ".npy"]
                else:
                    img += [img_fore_path + "test/" + position[i] + "/" + str(j) + "/" + img_post_path[k]]
                    label += [label_fore_path + position[i] + "/" + "obj_" + label_post_path[k] + str(j+count[i]) + ".npy"]
    #print(img)
    return img, label
    #if is_train:
    #    return ["/home/viplab/Downloads/frames/train/house/1/Lhand/"], ["/home/viplab/Downloads/labels/house/obj_left1.npy"]
    #else:
    #    return ["/home/viplab/Downloads/frames/test/house/1/Lhand/"], ["/home/viplab/Downloads/labels/house/obj_left4.npy"]


def file_num(is_train):
    image_path, _ = produce_path(is_train)

    copy = 1
    if is_train:
        copy = EPOCH
    image_filename_list = []
    for dir_path in image_path:
        file_list = os.listdir(dir_path)
        for tmp in file_list:
            image_filename_list = np.append(image_filename_list, dir_path + tmp)
    return len(image_filename_list)*copy






