# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:29:09 2017

@author: nvlab
"""

import tensorflow as tf
from time import time
import read_data
import numpy as np

trainEpoch = 50
batchSize = 64
totalBatchs = int(14992/batchSize)
learning_rate=0.00001
epoch_list=[]
obj_accuracy_list=[]
obj_loss_list=[]
ges_accuracy_list=[]
ges_loss_list=[]
FA_accuracy_list=[]
FA_loss_list=[]

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
C3_concate = tf.concat([C1_Pool, C3_Conv],3)
with tf.name_scope('C4_Conv'):
    W4 = weight([3,3,100,128])
    b4 = bias([128])
    Conv4 = conv2d(C3_concate, W4)+b4
    Conv4_BN = BN(Conv4, 128)
    C4_Conv = tf.nn.relu(Conv4_BN)    
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pooling(C4_Conv)
    
with tf.name_scope('C5_Conv'):
    W8 = weight([3,3,128,128])
    b8 = bias([128])
    Conv5 = conv2d(C2_Pool, W8)+b8
    Conv5_BN = BN(Conv5, 128)
    C5_Conv = tf.nn.relu(Conv5_BN)
C5_concate = tf.concat([C2_Pool, C5_Conv],3)
with tf.name_scope('C6_Conv'):
    W9 = weight([3,3,256,256])
    b9 = bias([256])
    Conv6 = conv2d(C5_concate, W9)+b9
    Conv6_BN = BN(Conv6, 256)
    C6_Conv = tf.nn.relu(Conv6_BN)    
with tf.name_scope('C3_Pool'):
    C3_Pool = max_pooling(C6_Conv)

with tf.name_scope('C7_Conv'):
    W10 = weight([3,3,256,256])
    b10 = bias([256])
    Conv7 = conv2d(C3_Pool, W10)+b10
    Conv7_BN = BN(Conv7, 256)
    C7_Conv = tf.nn.relu(Conv7_BN)
C7_concate = tf.concat([C3_Pool, C7_Conv],3)
with tf.name_scope('C8_Conv'):
    W11 = weight([3,3,512,512])
    b11 = bias([512])
    Conv8 = conv2d(C7_concate, W11)+b11
    Conv8_BN = BN(Conv8, 512)
    C8_Conv = tf.nn.relu(Conv8_BN)    
with tf.name_scope('C4_Pool'):
    C4_Pool = max_pooling(C8_Conv)
    
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C4_Pool, [-1,65536])
    
with tf.name_scope('D_Hidden1_Layer_obj'):
    W5 = weight([65536, 1024])
    b5 = bias([1024])
    D_Hidden1_obj = tf.matmul(D_Flat, W5)+b5
    D_Hidden1_obj_BN = BN(D_Hidden1_obj, 1024)
    D_Hidden1_obj_BN = tf.nn.relu(D_Hidden1_obj_BN)
    D_Hidden1_obj_Dropout = tf.nn.dropout(D_Hidden1_obj_BN, keep_prob=0.5)
with tf.name_scope('D_Hidden2_Layer_obj'):
    W6 = weight([1024, 512])
    b6 = bias([512])
    D_Hidden2_obj = tf.matmul(D_Hidden1_obj_Dropout, W6)+b6
    D_Hidden2_obj_BN = BN(D_Hidden2_obj, 512)
    D_Hidden2_obj_BN = tf.nn.relu(D_Hidden2_obj_BN)
    D_Hidden2_obj_Dropout = tf.nn.dropout(D_Hidden2_obj_BN, keep_prob=0.5)    
with tf.name_scope('Output_Layer_obj'):
    W7 = weight([512, 24])
    b7 = bias([24])
    y_obj_predict = tf.nn.softmax(tf.matmul(D_Hidden2_obj_Dropout, W7)+b7)    

with tf.name_scope('D_Hidden1_Layer_ges'):
    W10 = weight([65536, 1024])
    b10 = bias([1024])
    D_Hidden1_ges = tf.matmul(D_Flat, W10)+b10
    D_Hidden1_ges_BN = BN(D_Hidden1_ges, 1024)
    D_Hidden1_ges_BN = tf.nn.relu(D_Hidden1_ges_BN)
    D_Hidden1_ges_Dropout = tf.nn.dropout(D_Hidden1_ges_BN, keep_prob=0.5)
with tf.name_scope('D_Hidden2_Layer_ges'):
    W11 = weight([1024, 512])
    b11 = bias([512])
    D_Hidden2_ges = tf.matmul(D_Hidden1_ges_Dropout, W11)+b11
    D_Hidden2_ges_BN = BN(D_Hidden2_ges, 512)
    D_Hidden2_ges_BN = tf.nn.relu(D_Hidden2_ges_BN)
    D_Hidden2_ges_Dropout = tf.nn.dropout(D_Hidden2_ges_BN, keep_prob=0.5)    
with tf.name_scope('Output_Layer_ges'):
    W12 = weight([512,13])
    b12 = bias([13])
    y_ges_predict = tf.nn.softmax(tf.matmul(D_Hidden2_ges_Dropout, W12)+b12)    

with tf.name_scope('D_Hidden1_Layer_FA'):
    W13 = weight([65536, 1024])
    b13 = bias([1024])
    D_Hidden1_FA = tf.matmul(D_Flat, W13)+b13
    D_Hidden1_FA_BN = BN(D_Hidden1_FA, 1024)
    D_Hidden1_FA_BN = tf.nn.relu(D_Hidden1_FA_BN)
    D_Hidden1_FA_Dropout = tf.nn.dropout(D_Hidden1_FA_BN, keep_prob=0.5)
with tf.name_scope('D_Hidden2_Layer_FA'):
    W14 = weight([1024, 512])
    b14 = bias([512])
    D_Hidden2_FA = tf.matmul(D_Hidden1_FA_Dropout, W14)+b14
    D_Hidden2_FA_BN = BN(D_Hidden2_FA, 512)
    D_Hidden2_FA_BN = tf.nn.relu(D_Hidden2_FA_BN)
    D_Hidden2_FA_Dropout = tf.nn.dropout(D_Hidden2_FA_BN, keep_prob=0.5)    
with tf.name_scope('Output_Layer_FA'):
    W15 = weight([512, 2])
    b15 = bias([2])
    y_FA_predict = tf.nn.softmax(tf.matmul(D_Hidden2_FA_Dropout, W15)+b15)    

with tf.name_scope('optimizer'):
    y_label = tf.placeholder("float", shape=[None, 39],
                             name='y_label')
    y_obj_label = y_label[:,0:24]
    y_ges_label = y_label[:,24:37]
    y_FA_label = y_label[:,37:39]
 
    obj_loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                    (logits=y_obj_predict, labels=y_obj_label))
    ges_loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                    (logits=y_ges_predict, labels=y_ges_label))
    FA_loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                    (logits=y_FA_predict, labels=y_FA_label))
    loss_function = 0.6*obj_loss_function+0.3*ges_loss_function+0.1*FA_loss_function
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
    
with tf.name_scope('evaluate_model'):
    obj_correct_prediction = tf.equal(tf.argmax(y_obj_predict, 1),
                                  tf.argmax(y_obj_label, 1))
    obj_accuracy = tf.reduce_mean(tf.cast(obj_correct_prediction, "float"))
    ges_correct_prediction = tf.equal(tf.argmax(y_ges_predict, 1),
                                  tf.argmax(y_ges_label, 1))
    ges_accuracy = tf.reduce_mean(tf.cast(ges_correct_prediction, "float"))
    FA_correct_prediction = tf.equal(tf.argmax(y_FA_predict, 1),
                                  tf.argmax(y_FA_label, 1))
    FA_accuracy = tf.reduce_mean(tf.cast(FA_correct_prediction, "float"))
#    acc = tf.reduce_mean(tf.concat(obj_accuracy,ges_accuracy,FA_accuracy))
    
    
startTime = time()
saver = tf.train.Saver(tf.all_variables())
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    print("start training")
    n = 0
    for epoch in range(trainEpoch):
        for i in range(0, totalBatchs):
            images_x,train_label = read_data.read_data(batchSize)
            
            sess.run(optimizer, feed_dict={x:images_x, y_label:train_label})
            save_path = saver.save(sess, "./model.ckpt")

        if epoch >20:
            learning_rate = learning_rate*0.1
        
        test_x,test_label = read_data.read_data_test(batchSize)    
        obj_loss, obj_acc = sess.run([obj_loss_function, obj_accuracy],
                         feed_dict={x:test_x, y_label: test_label})
        ges_loss, ges_acc = sess.run([ges_loss_function, ges_accuracy],
                         feed_dict={x:test_x, y_label: test_label})
        FA_loss, FA_acc = sess.run([FA_loss_function, FA_accuracy],
                         feed_dict={x:test_x, y_label: test_label})
                             
        epoch_list.append(epoch)
        obj_loss_list.append(obj_loss)
        obj_accuracy_list.append(obj_acc)
        ges_loss_list.append(ges_loss)
        ges_accuracy_list.append(ges_acc)
        FA_loss_list.append(FA_loss)
        FA_accuracy_list.append(FA_acc)
        print("Train Epoch:", '%02d'%(epoch+1), 
              "Obj Loss=", "{:.9f}".format(obj_loss),"Obj Accuracy=","{:.9f}".format(obj_acc),
                "Ges Loss=", "{:.9f}".format(ges_loss),"Ges Accuracy=","{:.9f}".format(ges_acc),
                "FA Loss=", "{:.9f}".format(FA_loss),"FA Accuracy=","{:.9f}".format(FA_acc))
    sess.close()
          
duration = time()-startTime
print("Train Finished take:", duration)          