# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:31:28 2017

@author: user
"""
import tensorflow as tf
#import matplotlib.pyplot as plt 
import numpy as np
import input_data
import os

#train_dir = 'Lhand'
# define some constants
LR = 0.001                 # define learning rate
batch_size = 16             # define batch size
in_size = 200*200           # define input size
out_size = 24               # define output size
dropout = 0.5               # the probability to dropout
epoch = 50
train_data_dir = '/home/viplab/Desktop/petersci/CEDL_HW1/data/frames/train/'
label_dir = '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/'
test_data_dir = '/home/viplab/Desktop/petersci/CEDL_HW1/data/frames/test/'
save_dir = '/home/viplab/Desktop/petersci/CEDL_HW1/code/checkpoints/'
image_list, label_list = input_data.read_data(train_data_dir, label_dir, train = True)
test_image, test_label = input_data.read_data(test_data_dir, label_dir, train = False)
N_sample = len(image_list)

batch_xs, batch_ys = input_data.batch_generate(image_list, label_list, 200, 200, batch_size, 3000)
test_batch_xs, test_batch_ys =input_data.batch_generate(test_image, test_label, 200, 200, batch_size, 3000)
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 200, 200, 3])
ys = tf.placeholder(tf.float32, [None, out_size])
keep_prob = tf.placeholder(tf.float32)

# Alexnet structure
# conv_layer1
conv1 = tf.layers.conv2d(xs, filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu)
MaxPool1 = tf.layers.max_pooling2d(conv1, pool_size = 2, strides = 2)
norm1 = tf.nn.lrn(MaxPool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

# conv_layer2
conv2 = tf.layers.conv2d(norm1, 256, 3, 1, 'same', activation = tf.nn.relu)
MaxPool2 = tf.layers.max_pooling2d(conv2, pool_size = 2, strides = 2)
norm2 = tf.nn.lrn(MaxPool2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

# conv_layer3
conv3 = tf.layers.conv2d(norm2, 300, 3, 1, 'same', activation = tf.nn.relu)
#MaxPool3 = tf.layers.max_pooling2d(conv3, pool_size = 2, strides = 2)

# conv_layer4
conv4 = tf.layers.conv2d(conv3, 300, 3, 1, 'same', activation = tf.nn.relu)
#MaxPool4 = tf.layers.max_pooling2d(conv4, pool_size = 2, strides = 2)

# conv_layer5
conv5 = tf.layers.conv2d(conv4, 256, 3, 1, 'same', activation = tf.nn.relu)
MaxPool5 = tf.layers.max_pooling2d(conv5, pool_size = 2, strides = 2)
#norm5 = tf.nn.lrn(MaxPool5, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

# flatened
flat = tf.reshape(MaxPool5, [-1, 25*25*256])

# fully connected layer6
fc6 = tf.layers.dense(flat, 1024, activation = tf.nn.relu)
drop6 = tf.nn.dropout(fc6, dropout)

# fully connected layer7
fc7 = tf.layers.dense(drop6, 1024, activation = tf.nn.relu)
drop7 = tf.nn.dropout(fc7, dropout)

# fully connected layer8
output = tf.layers.dense(drop7, 24, activation = tf.nn.softmax)

# define loss function
loss = tf.losses.softmax_cross_entropy(onehot_labels = ys, logits = output)
train = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(ys, axis=1), tf.argmax(output, axis=1)), tf.float32))
t_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(ys, axis=1), tf.argmax(output, axis=1)), tf.float32))
#acc_correct = 0
saver = tf.train.Saver()

# start a session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)                                          # initialize all global variables
sess.run(tf.local_variables_initializer())              # initialize all local variables(such as acc_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

tot_count = 0
count = 0
compute_accuracy = 0
for i in range(epoch):
    for iteration in range(int(N_sample / batch_size)):
        if coord.should_stop():
            break
        count += 1
        tot_count += batch_size 
        image_batch, label_batch = sess.run([batch_xs, batch_ys])
        train_step = sess.run(train, feed_dict = {xs: image_batch, ys: label_batch})
        acc_correct = sess.run(accuracy, feed_dict = {xs: image_batch, ys: label_batch})
        compute_accuracy = acc_correct + compute_accuracy
        if iteration % 50 == 0:
            loss_step = sess.run(loss, feed_dict = {xs: image_batch, ys: label_batch})
            
            tot_accuracy = compute_accuracy / tot_count
            print('Step:', count, '| train loss: %.4f' % loss_step, '| train accuracy: %.2f' % tot_accuracy)
        if (i == epoch - 1) and (iteration == int(N_sample / batch_size) - 1):
            checkpoint_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=count)
    


tot_count = 0
count = 0
test_accuracy = 0
tot_test_accuracy = 0
for step in range(int(len(test_image) / batch_size)):
    if coord.should_stop():
        break
    count += 1
    tot_count += batch_size
    test_batch_x, test_batch_y = sess.run([test_batch_xs, test_batch_ys])
    acc_ = sess.run(t_accuracy, feed_dict = {xs: test_batch_x, ys: test_batch_y})
    test_accuracy = acc_ + test_accuracy
    tot_test_accuracy = test_accuracy / tot_count
print('test accuracy: %.2f' % tot_test_accuracy)

coord.request_stop()       
coord.join(threads)
sess.close()

