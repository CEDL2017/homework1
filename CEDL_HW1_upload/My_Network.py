# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:23:12 2017

@author: PC-Tong
"""

import tensorflow as tf

import util

OUTPUT_SIZE = 24
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 108
SAVE_MODEL_PATH = "./model_log/"
MODEL_FILENAME = "model.ckpt"


class Mynetwork:
    
    def _init_(self):
        pass
    
    # build weights
    def build_weight(self, shape, name):  #shape = [kernel_width, kernek_length, input_size, output_size]
        return tf.Variable(tf.truncated_normal(shape = shape, stddev = .1))
        #tmp = tf.contrib.layers.xavier_initializer()
        #return tf.get_variable(name = name, shape = shape, dtype = tf.float32, 
        #                          initializer=tmp)
    
    # build biases
    def build_bias(self, shape):  #shape = 1*1 list
        return tf.Variable(tf.constant(.1, shape = shape))
        
    #build convolution layers
    def conv_layer(self, input_image, shape, name):
        weight = self.build_weight(shape, name)
        bias = self.build_bias([shape[3]])
        #print(input_image.shape)
        #print(weight.shape)
        conv_val = tf.nn.bias_add(tf.nn.conv2d(input_image, weight, strides = [1, 1, 1, 1], padding = 'SAME'), bias)
        
        return conv_val
    
    # build a max pooling layers with size 2*2
    def max_pool_2_2(self, input_image):
        return tf.nn.max_pool(input_image, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # build a fully connected layers
    def fc_layer(self, input_image, size, name):
        shape = input_image.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        input_flat= tf.reshape(input_image, [-1, dim])
        weight = self.build_weight([dim, size], name)
        bias = self.build_bias([size])
        result = tf.nn.bias_add(tf.matmul(input_flat, weight), bias)
        
        return result
    
    # model of VGG-16
    def model(self, image):
        # model structures
        #reshape_image = tf.reshape(image, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        #print(image.shape)
        # conv 1
        self.conv_1_1 = self.conv_layer(image, [3, 3, 3, 4], "conv1_1")
        self.relu_1_1 = tf.nn.relu(self.conv_1_1)
        self.conv_1_2 = self.conv_layer(self.relu_1_1, [3, 3, 4, 4], "conv1_2")
        self.relu_1_2 = tf.nn.relu(self.conv_1_2)
        self.pool1 = self.max_pool_2_2(self.relu_1_2)
        
        # conv2
        self.conv_2_1 = self.conv_layer(self.pool1, [3, 3, 4, 8], "conv2_1")
        self.relu_2_1 = tf.nn.relu(self.conv_2_1)
        self.conv_2_2 = self.conv_layer(self.relu_2_1, [3, 3, 8, 8], "conv2_2")
        self.relu_2_2 = tf.nn.relu(self.conv_2_2)
        self.pool2 = self.max_pool_2_2(self.relu_2_2)
        
        # conv3
        self.conv_3_1 = self.conv_layer(self.pool2, [3, 3, 8, 16], "conv3_1")
        self.relu_3_1 = tf.nn.relu(self.conv_3_1)
        self.conv_3_2 = self.conv_layer(self.relu_3_1, [3, 3, 16, 16], "conv3_2")
        self.relu_3_2 = tf.nn.relu(self.conv_3_2)
        self.conv_3_3 = self.conv_layer(self.relu_3_2, [3, 3, 16, 16], "conv3_3")
        self.relu_3_3 = tf.nn.relu(self.conv_3_3)
        self.pool3 = self.max_pool_2_2(self.relu_3_3)
        
        # conv4
        self.conv_4_1 = self.conv_layer(self.pool3, [3, 3, 16, 32], "conv4_1")
        self.relu_4_1 = tf.nn.relu(self.conv_4_1)
        self.conv_4_2 = self.conv_layer(self.relu_4_1, [3, 3, 32, 32], "conv4_2")
        self.relu_4_2 = tf.nn.relu(self.conv_4_2)
        self.conv_4_3 = self.conv_layer(self.relu_4_2, [3, 3, 32, 32], "conv4_3")
        self.relu_4_3 = tf.nn.relu(self.conv_4_3)
        self.pool4 = self.max_pool_2_2(self.relu_4_3)
        
        # conv5
        self.conv_5_1 = self.conv_layer(self.pool4, [3, 3, 32, 64], "conv5_1")
        self.relu_5_1 = tf.nn.relu(self.conv_5_1)
        self.conv_5_2 = self.conv_layer(self.relu_5_1, [3, 3, 64, 64], "conv5_2")
        self.relu_5_2 = tf.nn.relu(self.conv_5_2)
        self.conv_5_3 = self.conv_layer(self.relu_5_2, [3, 3, 64, 64], "conv5_3")
        self.relu_5_3 = tf.nn.relu(self.conv_5_3)
        self.pool5 = self.max_pool_2_2(self.relu_5_3)
        
        # fc6
        self.fc6 = self.fc_layer(self.pool5, 512, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        
        # fc7
        self.fc7 = self.fc_layer(self.relu6, 1024, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        
        # fc8
        self.fc8 = self.fc_layer(self.relu7, OUTPUT_SIZE, "fc8")
        self.prediction = tf.nn.softmax(self.fc8)
        
        return self.prediction

    def loss_func(self, prediction, label):
        #self.loss = tf.reduce_mean(-tf.reduce_sum(label * tf.log(prediction),
        #                                              reduction_indices=[1]))
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label, prediction))
        return self.loss
    

    def train(self, rate, loss):
        
        global_step = tf.Variable(0, trainable = False)
        decay_step = .5
        decay_rate = 1
        learning_rate = tf.train.inverse_time_decay(rate, global_step,
                                                        decay_step, decay_rate)
        return tf.train.AdamOptimizer(rate).minimize(loss)


    def save_model(self, sess):
        saver = tf.train.Saver()
        return saver.save(sess, SAVE_MODEL_PATH + MODEL_FILENAME)


    def load_model(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_MODEL_PATH + MODEL_FILENAME)
        sess.run(tf.global_variables())

    def count_correct(self, prediction, label):
        equality = 0
        equality = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
        #print("-----------------------------------------------------")
        #print(equality)
        total_equal = tf.reduce_sum(tf.cast(equality, tf.float32))
        return total_equal
        #return 1
