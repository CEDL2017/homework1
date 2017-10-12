# jack841021's homework 1: image classification

## Overview

The project is related to:

Karen Simonyan, Andrew Zisserman: [1409.1556] Very Deep Convolutional Networks for Large-Scale Image Recognition

However, I've made quite some modifications.

## Implementation

### Image processing:

3 channels => 1 channel (grey scale), color is meaningless
 
1080 * 1920 => 128 * 228

### Lables

one-hot encoding

### Neural network

(convolution + elu + maxpooling) * 6 + fullyconnected

The reason I build a model like is that I want to create an "information bottleneck".

If you're interested, please refer to:

Naftali Tishby, Noga Zaslavsky: [1503.02406] Deep Learning and the Information Bottleneck Principle

Therefore, The first six layers are used as a bottleneck and the last one is a classifier.

### Activation function

ELU outperformed RELU, which I think may due to dead neurons created by RELU.

### Optimization

Stochastic gradient descent

In my experience, sgd with more than 0.9 momentum usually does pretty well. The only drawback is the training time.

## Required libraries

numpy

scipy

keras and its requirements

### Results

https://www.dropbox.com/sh/p4av3keskz0bxjq/AABu-1y25p5lahixQg81528Ga?dl=0
