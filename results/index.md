# Kuanwei Ho<span style="color:red">(103062134)</span>

#Project 5: Deep Classification

## Overview
The project is related to accomplish image classification task in transfer learning method.
This project concentrates on using pretrained ImageNet model such as Inception-Resnet-v2 and vgg-16 released by Tensorflow and fine tune them to precisely classify handcam's photo to detect 24 classes of objects.

The best result testing on total test data of fine tuning each model:
### Inceptino-Resnet-v2: 
	streaming accuracy: 50%
### vgg-16:
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

I've conducted experiment on two different models: **Inception-Resnet and vgg16**.
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

```Python
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(batch_image, 
                                                 num_classes = num_classes)
	
    exclude = ['InceptionResnetV2/AuxLogits',
               'InceptionResnetV2/Logits']
    variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
```

I've tried the **Adam optimizer, Momentum optimizer**(momentum = 0.9, 0.95), **RMSProp optimizer**, learning rate starts at 0.045, 0.001, 0.0002 with decay 0.7/0.8/0.9 per 2 epochs.
Inception model also requires certain preprocessing that every pixel in the image should be (-1, 1).
Moreover, Tensorflow offers a way to 

The loss hardly decreases, and the accuracy converges to nearly 50%.

In my opinion, this problem might due to some reasons:
+ The weights aren't properly initialized or trained
	There might be some mistakes made by my coding error which results in the bad performance.
	For example, the code should only initialize layers not to be trained, and train only specific scopes(Logits, AuxLogits).
	However, the performance seems that I train the model from scratch.
	The model hardly extracts any feature from images.
	
+ Inproper paramater tuning
	The model always reaches to same convergence no matter what the parameter was set.
	Although I've tried 3 different optimizers with 3 sets of decaying learning rate, it seems that the model is still trapped in a local minimum.

	Maybe there is still other way to break this deadlock.
+ The model wasn't suitable for the task
	Inceprtion-Resent-v2 is a model requires lots of hard-restricted 

### 2. vgg-16



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


