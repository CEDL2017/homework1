"""
This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all 
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Following my blogpost at:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of classes.
The structure of this script is strongly inspired by the fast.ai Deep Learning
class by Jeremy Howard and Rachel Thomas, especially their vgg16 finetuning
script:  
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same folder: 
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/  

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np

class SimpleNN(object):
  
  def __init__(self, x, keep_prob, num_classes,
               weights_path = 'DEFAULT'):
    

    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    
    # if weights_path == 'DEFAULT':      
    #   self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
    # else:
    #   self.WEIGHTS_PATH = weights_path
    
    # Call the create function to build the computational graph of AlexNet
    self.create()
    
  def create(self):
    

    # # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
    # conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
    # pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
    # norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
    
    # # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
    # conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
    # pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
    # norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
    
    # # 3rd Layer: Conv (w ReLu)
    # conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')
    
    # # 4th Layer: Conv (w ReLu) splitted into two groups
    # conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')
    
    # # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    # conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
    # pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')
    
    # # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    # flattened = tf.reshape(pool5, [-1, 6*6*256])
    # fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
    # dropout6 = dropout(fc6, self.KEEP_PROB)
    
    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(self.X, 8192, 4096, name = 'fc7')
    dropout7 = dropout(fc7, self.KEEP_PROB)
    
    # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')
     
  
"""
Predefine all necessary layer for the AlexNet
""" 
  
def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    try:
      # Create tf variables for the weights and biases
      weights = tf.get_variable('weights_nn', shape=[num_in, num_out], trainable=True)
      biases = tf.get_variable('biases_nn', [num_out], trainable=True)
    except ValueError:
      scope.reuse_variables()
      weights = tf.get_variable('weights_nn')
      biases = tf.get_variable('biases_nn')
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)
      return relu
    else:
      return act
     
def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)
  
    