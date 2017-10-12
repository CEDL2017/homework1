"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import utils
import numpy as np
import utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import cv2
import os
import pdb
import random
import time
num_classes = 24
batch_size = 32
num_epochs = 80
lr = 1e-4
training_mode = 0#True

### read data ###
train_images_root = '../frames/train'
test_images_root = '../frames/test'
label_root = '../labels'


data = []
for locat in os.listdir(train_images_root):
    for pool in os.listdir(os.path.join(train_images_root, locat)):
        for view in os.listdir(os.path.join(train_images_root, locat, pool)):
            if view in ['Lhand','Rhand']:
                if view == 'Lhand':
                    label_array = np.load(os.path.join(label_root,locat,'obj_left'+ pool+'.npy'))
                elif view == 'Rhand':
                    label_array = np.load(os.path.join(label_root,locat,'obj_right'+ pool+'.npy'))
                for img_name in os.listdir(os.path.join(train_images_root, locat, pool, view)):
                    idx = int(img_name.split('.png')[0].split('Image')[1])-1
                    target = label_array[idx]
                    img_path = os.path.join(train_images_root, locat, pool, view,img_name)
                    data.append([img_path,target])
data_test = []
for locat in os.listdir(test_images_root):
    for pool in os.listdir(os.path.join(test_images_root, locat)):
        for view in os.listdir(os.path.join(test_images_root, locat, pool)):
            if view in ['Lhand','Rhand']:
                if view == 'Lhand':
                    if locat == 'lab':     
                        label_array = np.load(os.path.join(label_root,locat,'obj_left'+ str(int(pool)+4)+'.npy'))
                    else:    
                        label_array = np.load(os.path.join(label_root,locat,'obj_left'+ str(int(pool)+3)+'.npy'))

                elif view == 'Rhand':
                    if locat == 'lab':     
                        label_array = np.load(os.path.join(label_root,locat,'obj_right'+ str(int(pool)+4)+'.npy'))
                    else:    
                        label_array = np.load(os.path.join(label_root,locat,'obj_right'+ str(int(pool)+3)+'.npy'))
                for img_name in os.listdir(os.path.join(test_images_root, locat, pool, view)):
                    idx = int(img_name.split('.png')[0].split('Image')[1])-1
                    target = label_array[idx]
                    img_path = os.path.join(test_images_root, locat, pool, view,img_name)
                    #img = cv2.resize(cv2.imread(img_path),(224,224))
                    data_test.append([img_path,target])


#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.device('/gpu:0'):
    sess = tf.Session()#config=config)

    images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [batch_size, num_classes])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = true_out, logits = vgg.fc8) 
    loss = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer(lr,beta1=0.8).minimize(loss)
    #train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
            
    sess.run(tf.global_variables_initializer())
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(max_to_keep=10)
        saver.restore(sess, "models/my-model-16000")
    # training
    iter_times = 0
    if training_mode:    
        for epo in range(num_epochs):
                    
            print('time', time.time())
            random.shuffle(data)
            random.shuffle(data_test)
            for b in range(int(np.floor(len(data)/batch_size))):   
                batch_imgs = []
                batch_labels = []
                for count in range(batch_size):
                    img_path = data[b*batch_size+count][0]
                    img = cv2.resize(cv2.imread(img_path),(224,224))
                    batch_imgs.append(img)
                    label = np.zeros(num_classes).astype(np.int32); 
                    label[int(data[b*batch_size+count][1])] =1 ;
                    batch_labels.append(label)
                batch_imgs = np.asarray(batch_imgs)
                batch_labels = np.asarray(batch_labels)
                
                _loss, _prob, _ = sess.run([loss, vgg.prob, train_op], feed_dict={images: batch_imgs, true_out: batch_labels, train_mode: True })
                train_accu = (_prob.argmax(1) == batch_labels.argmax(1)).astype(np.float32).mean()
                iter_times += 1
                if not iter_times % 20:
                    print("iter: %d\tloss: %.5f, accuracy: %.4f")%(iter_times, _loss, train_accu)
                if not iter_times % 2000:
                    saver.save(sess, './models/my-model', global_step=(iter_times))    
            
            # test classification every n epoch
            if not epo % 2:
                accu = 0.
                for tb in range(15): #range(int(np.floor(len(data_test)/batch_size))): 
                    batch_imgs = []
                    batch_labels = []
                    for count in range(batch_size):
                        img_path = data_test[tb*batch_size+count][0]
                        img = cv2.resize(cv2.imread(img_path),(224,224))
                        batch_imgs.append(img)
                        label = np.zeros(num_classes).astype(np.int32); 
                        label[int(data_test[tb*batch_size+count][1])] =1 ;
                        batch_labels.append(label)
                    batch_imgs = np.asarray(batch_imgs)
                    batch_labels = np.asarray(batch_labels)
                    
                    prob = sess.run(vgg.prob, feed_dict={images: batch_imgs, train_mode: False})
                    accu += (prob.argmax(1) == batch_labels.argmax(1)).astype(np.float32).mean()
                accu =accu / 15.
                print('epoch %d(%d iters), accuracy: %.4f')%(epo,iter_times,accu)
    
    ### testing ###
    accu = 0.
    for tb in range(int(np.floor(len(data_test)/batch_size))): 
        batch_imgs = []
        batch_labels = []
        for count in range(batch_size):
            img_path = data_test[tb*batch_size+count][0]
            img = cv2.resize(cv2.imread(img_path),(224,224))
            batch_imgs.append(img)
            label = np.zeros(num_classes).astype(np.int32); 
            label[int(data_test[tb*batch_size+count][1])] =1 ;
            batch_labels.append(label)
        batch_imgs = np.asarray(batch_imgs)
        batch_labels = np.asarray(batch_labels)
        
        prob = sess.run(vgg.prob, feed_dict={images: batch_imgs, train_mode: False})
        
        accu += (prob.argmax(1) == batch_labels.argmax(1)).astype(np.float32).sum()
        sub_accu = (prob.argmax(1) == batch_labels.argmax(1)).astype(np.float32).mean()
        #accu += sub_accu
        print('test batch %d, accuracy: %.4f')%(tb, sub_accu)
    accu = float(accu) / len(data_test) #(len(data_test)/batch_size)
    print('***Testing***, accuracy: %.4f')%(accu)
         

