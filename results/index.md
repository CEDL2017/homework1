# Kuanwei Ho<span style="color:red">(103062134)</span>

#Project 5: Deep Classification

## Overview
The project is related to accomplish image classification task in transfer learning method.
This project concentrates on using pretrained ImageNet model such as Inception-Resnet-v2 and vgg-16 released by Tensorflow and fine tune them to precisely classify handcam's photo to detect 24 classes of objects.

The best result testing on total test data of fine tuning each model:
#### Inceptino-Resnet-v2: 
	streaming accuracy: 50%
#### vgg-16:
	streaming accuracy: 
	

### Reference:
- Thesis:
	[Recognition from Hand Cameras: A Revisit with Deep Learning](https://arxiv.org/abs/1512.01881)
	[Multi-label Classification of Satellite Images with Deep Learning](http://cs231n.stanford.edu/reports/2017/pdfs/908.pdf)
- Tutorial:
	[kwotsin/transfer_learning_tutorial](https://github.com/kwotsin/transfer_learning_tutorial)
	[kratzert/finetune_alexnet_with_tensorflow](https://github.com/kratzert/finetune_alexnet_with_tensorflow)
	[TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)

	


## Implementation

I've conducted experiment on two different models: Inception-Resnet and vgg16.
The performance of Inception-Resnet is very depressing, and vgg16 is easier to train.
I'll discuss them repsectively in the following.


### 1. Inception-Resnet-v2

![](https://i2.kknews.cc/SIG=e19v49/q4q0006048n3505os81.jpg)

The model and specific source code are released by Tensorflow and implemented in tf-slim, which is a package helping people to much more easily train model from scratch or pretrained resources.
The implementation is in **inception_resnet_trainer.py**.

The py-file firstly do the data preparing.
It loads labels data into program and process them and train images as tf.FIFOQueue for tf.train.batch.

The code following data preparation is about loading Inception-Resnet architecture which is provided by tf-slim's **inception-resnet.py** with total nearly 500 trainable variables.
Then, we only train the specific layers we want, which are the most outside layers in the figure above.
They are Logit and Auxiliary Logit scope's layer, which is in charge of the final output of model.

I've tried the Adam optimizer, Momentum optimizer, learning rate starts at 0.045, 0.001, 0.0002 with decay 0.7/0.8/0.9 per 2 epochs.
The loss is 
	

### 2. vgg-16

```
Code highlights
```

## Installation
* Other required packages.
* How to compile from source?

### Results

<table border=1>
<tr>
<td>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

<tr>
<td>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

</table>


