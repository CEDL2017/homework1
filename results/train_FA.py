# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:29:09 2017

@author: nvlab
"""

import tensorflow as tf
from time import time
import read_data_FA
import numpy as np

trainEpoch = 30
batchSize = 64
totalBatchs = int(14992/batchSize)
learning_rate=0.00001
epoch_list=[]
accuracy_list=[]
loss_list=[]


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), 
                       name='W', dtype=tf.float32)                    
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b',
                       dtype=tf.float32)    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')    
def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding='SAME')
def BN(Conv, out_size):
    fc_mean, fc_var = tf.nn.moments(Conv, axes=[0])
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon=0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)
    mean, var = mean_var_with_update()
    Conv = tf.nn.batch_normalization(Conv, mean, var, shift, scale, epsilon)
    return Conv
                          
with tf.name_scope('Input_layer'):
    x = tf.placeholder("float", shape=[None, 95232])
    x_image = tf.reshape(x, [-1, 128,248, 3])
    
with tf.name_scope('C1_Conv'):
    W1 = weight([3,3,3,36])
    b1 = bias([36])
    Conv1 = conv2d(x_image, W1)+b1
    Conv1_BN = BN(Conv1, 36)
    C1_Conv = tf.nn.relu(Conv1_BN)  
with tf.name_scope('C2_Conv'):
    W2 = weight([3,3,36,36])
    b2 = bias([36])
    Conv2 = conv2d(C1_Conv, W2)+b2
    Conv2_BN = BN(Conv2, 36)
    C2_Conv = tf.nn.relu(Conv2_BN) 
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pooling(C2_Conv)
    
with tf.name_scope('C3_Conv'):
    W3 = weight([3,3,36,64])
    b3 = bias([64])
    Conv3 = conv2d(C1_Pool, W3)+b3
    Conv3_BN = BN(Conv3, 64)
    C3_Conv = tf.nn.relu(Conv3_BN)
with tf.name_scope('C4_Conv'):
    W4 = weight([3,3,64,64])
    b4 = bias([64])
    Conv4 = conv2d(C3_Conv, W4)+b4
    Conv4_BN = BN(Conv4, 64)
    C4_Conv = tf.nn.relu(Conv4_BN)    
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pooling(C4_Conv)
    
with tf.name_scope('C5_Conv'):
    W8 = weight([3,3,64,128])
    b8 = bias([128])
    Conv5 = conv2d(C2_Pool, W8)+b8
    Conv5_BN = BN(Conv5, 128)
    C5_Conv = tf.nn.relu(Conv5_BN)
with tf.name_scope('C6_Conv'):
    W9 = weight([3,3,128,128])
    b9 = bias([128])
    Conv6 = conv2d(C5_Conv, W9)+b9
    Conv6_BN = BN(Conv6, 128)
    C6_Conv = tf.nn.relu(Conv6_BN)    
with tf.name_scope('C3_Pool'):
    C3_Pool = max_pooling(C6_Conv)
    
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C3_Pool, [-1,63488])
    
with tf.name_scope('D_Hidden_Layer'):
    W5 = weight([63488, 1024])
    b5 = bias([1024])
    D_Hidden1 = tf.matmul(D_Flat, W5)+b5
    D_Hidden1_BN = BN(D_Hidden1, 1024)
    D_Hidden1_BN = tf.nn.relu(D_Hidden1_BN)
    D_Hidden1_Dropout = tf.nn.dropout(D_Hidden1_BN, keep_prob=0.5)
with tf.name_scope('D_Hidden_Layer'):
    W6 = weight([1024, 512])
    b6 = bias([512])
    D_Hidden2 = tf.matmul(D_Hidden1_Dropout, W6)+b6
    D_Hidden2_BN = BN(D_Hidden2, 512)
    D_Hidden2_BN = tf.nn.relu(D_Hidden2_BN)
    D_Hidden2_Dropout = tf.nn.dropout(D_Hidden2_BN, keep_prob=0.5)    
with tf.name_scope('Output_Layer'):
    W7 = weight([512, 2])
    b7 = bias([2])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden2_Dropout, W7)+b7)    

with tf.name_scope('optimizer'):
    y_label = tf.placeholder("float", shape=[None, 2],
                             name='y_label')
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                    (logits=y_predict, labels=y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
    
with tf.name_scope('evaluate_model'):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1),
                                  tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


startTime = time()
saver = tf.train.Saver(tf.all_variables())
f = open('FA.txt','a')
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    print("start training")
    for epoch in range(trainEpoch):
        for i in range(0, totalBatchs):
            images_x, batch_y = read_data_FA.read_data(batchSize)
           
            sess.run(optimizer, feed_dict={x:images_x, y_label:batch_y})
            save_path = saver.save(sess, "./model/model_FA.ckpt")

        if epoch >20:
            learning_rate = learning_rate*0.1
        
        val_x, test_y = read_data_FA.read_data_test(batchSize)    
        loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x:val_x, y_label: test_y})
                             
#        epoch_list.append(epoch)
#        loss_list.append(loss)
#        accuracy_list.append(acc)
        f.write(str(epoch)+"\n")
        f.write(str(loss)+"\n")
        f.write(str(acc)+"\n")
        print("Train Epoch:", '%02d'%(epoch+1), 
              "Loss=", "{:.9f}".format(loss),
              "Accuracy=", acc)
    sess.close()
          
duration = time()-startTime
print("Train Finished take:", duration)          