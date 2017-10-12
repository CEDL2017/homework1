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

ITERATION = 100



pre_process = False
if "--pre" in sys.argv:
	pre_process = True
is_train = False
if "--train" in sys.argv:
	is_train = True
elif "--test" in sys.argv:
	is_train = False

if pre_process:
	image_folder_list, label_folder_list = util.produce_path(is_train)
	util.pre_process(image_folder_list, label_folder_list)
print('pre-preocess done!!')

images, labels = util.load_file()

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

my_net = My_Network.Mynetwork()
training = my_net.train(images, labels)

# run training
with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    if is_train:
    	for i in range(ITERATION):
    		img, lab = sess.run([images, labels])
    		print("~!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!")
    		val = sess.run(training)
    		print(val)
    	my_net.save_model(sess)
    else:
    	pass
    coord.request_stop()
    coord.join(threads)

