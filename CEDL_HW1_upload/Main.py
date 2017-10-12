# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:35:21 2017

@author: PC-Tong
"""

"""
Main file
"""

import tensorflow as tf
import sys
import os

import My_Network
import util

LEARNING_RATE = 1e-5



pre_process = False
if "--pre" in sys.argv:
    pre_process = True
is_train = False
if "--train" in sys.argv:
    is_train = True
elif "--test" in sys.argv:
    is_train = False

if pre_process:
    util.pre_process(is_train)
print('pre-preocess done!!')

images, labels = util.load_file(is_train)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

xs = tf.placeholder(tf.float32, shape = [util.BATCH_SIZE, util.IMAGE_HEIGHT, util.IMAGE_WIDTH, 3])
ys = tf.placeholder(tf.float32, shape = [util.BATCH_SIZE, My_Network.OUTPUT_SIZE])

my_net = My_Network.Mynetwork()
prediction = my_net.model(xs)
loss = my_net.loss_func(prediction, ys)
correct = my_net.count_correct(prediction, ys)
training = my_net.train(LEARNING_RATE, loss)

# run training
with tf.Session() as sess:
    #sess.run(init_op)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    

    total_correct = 0
    total_count = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    if is_train:
        print(int(util.file_num(is_train)/util.BATCH_SIZE))
        for i in range(int(util.file_num(is_train)/util.BATCH_SIZE)):
            img, lab = sess.run([images, labels])
            #print(img[0][0])
            #print("~!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!")
            _, p, l, c = sess.run([training, prediction, loss, correct], feed_dict = {xs:img, ys:lab})
            total_correct += c
            total_count += util.BATCH_SIZE
            print("iteration:", i, "loss:", l, "accuracy", total_correct/total_count)
        my_net.save_model(sess)
    else:
        #tf.reset_default_graph()
        my_net.load_model(sess)
        loop_num = int(util.file_num(is_train)/util.BATCH_SIZE)
        for i in range(loop_num):
            img, lab = sess.run([images, labels])
            l, c = sess.run([loss, correct], feed_dict = {xs:img, ys:lab})
            #print(c)
            total_correct += c
            total_count += util.BATCH_SIZE
            print("iteration:", i, "lost:", l, "accuracy", total_correct/total_count)


    coord.request_stop()
    coord.join(threads)

