# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:39:16 2017

@author: nvlab
"""

import tensorflow as tf
from time import time
import read_data_obj
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle

batchSize = 64
trainEpoch = 12776/batchSize
learning_rate=0.00001
epoch_list=[]
accuracy_list=[]
loss_list=[]
precision = dict()
recall = dict()
average_precision = dict()

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
    
with tf.name_scope('D_Hidden1_Layer'):
    W5 = weight([65536, 1024])
    b5 = bias([1024])
    D_Hidden1 = tf.matmul(D_Flat, W5)+b5
    D_Hidden1_BN = BN(D_Hidden1, 1024)
    D_Hidden1_BN = tf.nn.relu(D_Hidden1_BN)
    D_Hidden1_Dropout = tf.nn.dropout(D_Hidden1_BN, keep_prob=0.5)
with tf.name_scope('D_Hidden2_Layer'):
    W6 = weight([1024, 512])
    b6 = bias([512])
    D_Hidden2 = tf.matmul(D_Hidden1_Dropout, W6)+b6
    D_Hidden2_BN = BN(D_Hidden2, 512)
    D_Hidden2_BN = tf.nn.relu(D_Hidden2_BN)
    D_Hidden2_Dropout = tf.nn.dropout(D_Hidden2_BN, keep_prob=0.5)    
with tf.name_scope('Output_Layer'):
    W7 = weight([512, 24])
    b7 = bias([24])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden2_Dropout, W7)+b7)    

with tf.name_scope('optimizer'):
    y_label = tf.placeholder("float", shape=[None, 24],
                             name='y_label')
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                    (logits=y_predict, labels=y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
    
with tf.name_scope('evaluate_model'):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1),
                                  tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "model/model_obj.ckpt")
    print('Model restored')
    for k in range(trainEpoch):   
        if k ==0:        
            val_x = read_data_obj.read_data_test_(batchSize)    
            y_test_all = sess.run(y_predict, feed_dict={x:val_x})
        else:
            val_x = read_data_obj.read_data_test_(batchSize)    
            y_test = sess.run(y_predict, feed_dict={x:val_x})
            y_test_all = np.vstack((y_test_all,y_test))
    y_test_all = np.array(y_test_all)
    y_score = read_data_obj.read_obj_labels_test()
    for i in range(24):
        precision[i], recall[i], _ = precision_recall_curve(y_score[:, i],
                                                        y_test_all[:, i])
        average_precision[i] = average_precision_score(y_score[:, i], y_test_all[:, i])
            
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_score.ravel(),
            y_test_all.ravel())
    average_precision["micro"] = average_precision_score(y_score,y_test_all,
                                                     average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
            .format(average_precision["micro"]))
            
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
     where='post')
    plt.fill_between(recall["micro"], precision["micro"],alpha=0.2,color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AUC={0:0.2f}'
            .format(average_precision["micro"]))
            
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

    for i, color in zip(range(24), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of obj Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0.55, 0.55), prop=dict(size=8))

    
    sess.close()
    
plt.savefig('obj_precision-recall.png')
    
                   