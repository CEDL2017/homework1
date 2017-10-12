# Kuanwei Ho<span style="color:red">(103062134)</span>

#Project 5: Deep Classification

## Overview
The project is related to accomplish image classification task in transfer learning method.<br /> 
This project concentrates on using pretrained ImageNet model such as Inception-Resnet-v2 and vgg-16 released by Tensorflow and fine tune them to precisely classify handcam's photo to detect 24 classes of objects.

The best result testing on total test data of fine tuning each model:
### Inception-Resnet-v2: 
	streaming accuracy: 50%
### vgg-16:
	streaming accuracy: 

### Reference:
- Thesis:

	[Recognition from Hand Cameras: A Revisit with Deep Learning](https://arxiv.org/abs/1512.01881)<br />
	[Multi-label Classification of Satellite Images with Deep Learning](http://cs231n.stanford.edu/reports/2017/pdfs/908.pdf)
- Resource:

	[kwotsin/transfer_learning_tutorial](https://github.com/kwotsin/transfer_learning_tutorial)<br />
	[kratzert/finetune_alexnet_with_tensorflow](https://github.com/kratzert/finetune_alexnet_with_tensorflow)<br />
	[TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)<br />

## Implementation

I've conducted experiment on two different models: **Inception-Resnet and vgg16**.<br />
The performance of Inception-Resnet is very depressing, and vgg16 is easier to train.<br />
I'll discuss them repsectively in the following.<br />


### 1. Inception-Resnet-v2

![](https://i2.kknews.cc/SIG=e19v49/q4q0006048n3505os81.jpg)

The model and specific source code are released by Tensorflow and implemented in **tf-slim**, which is a package helping people to much more easily train model from scratch or pretrained resources.<br />
The implementation is in **inception_resnet_trainer.py**.

The py-file firstly do the data preparing.<br />
It loads labels data into program and process them and train images as QueueRunner for tf.train.batch.<br />

The code following data preparation is about loading Inception-Resnet architecture which is provided by tf-slim's **inception-resnet.py** with total nearly 500 trainable variables.<br />
Then, we only train the specific layers we want, which are layers in the tail of the figure above. <br />
They are Logit and Auxiliary Logit scope's layers, which is in charge of the final output of model. <br />
 
```Python
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(batch_image, 
                                                 num_classes = num_classes)
	
    exclude = ['InceptionResnetV2/AuxLogits',
               'InceptionResnetV2/Logits']
    variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
```
Moreover, Tensorflow offers a way to do preprocessing, and this preprocessing method also does data augmentation on the training image.

I've tried the **Adam optimizer, Momentum optimizer**(momentum = 0.9, 0.95), **RMSProp optimizer**, learning rate starts at 0.045, 0.001, 0.0002 with decay 0.7/0.8/0.9 per 2 epochs.<br />
Inception model also requires certain preprocessing that every pixel in the image should be (-1, 1).<br />
The loss function is **softmax cross entropy**.
The batch size I've tried 1, 4, 8, 16 and 32. The number of epoch I've tried 


However, the loss hardly decreases, and the accuracy converges only to nearly 50%
In my opinion, this problem might due to some reasons:<br />
+ **The weights aren't properly initialized or trained<br />**
	There might be some mistakes made by my coding error which results in the bad performance.
	For example, the code should only initialize layers not to be trained, and train only specific scopes(Logits, AuxLogits).<br />
	However, the performance seems that I train the model from scratch.<br />
	The model hardly extracts any feature from images.<br />
	
+ **Inproper paramater tuning<br />**
	The model always reaches to same convergence no matter what the parameter was set.<br />
	Although I've tried 3 different optimizers with 3 sets of decaying learning rate, it seems that the model is still trapped in a local minimum.

	Maybe there is still other way to break this deadlock.<br />
+ **The model wasn't suitable for the task<br />**
	Inceprtion-Resent-v2 is a very complex model, it's feature extraction might not be able to catch features in handcam's photo.<br />
	There are also possibilites that the choices of which layers to be trained is wrong. <br />
	Maybe too much or too less layers to be altered.<br />
	I've tried to train ConvNet variables before Logits and AuxLogits, but the performance is still the same.

### 2. vgg-16

![](https://www.cs.toronto.edu/~frossard/post/vgg16/vgg16.png)

As a result, vgg-16 is a model more easier to do tranfer learning.

```Python
    logits, end_points = vgg.vgg_16(batch_image, 
                                    num_classes = num_classes,
                                    is_training = False)
    variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8'])
```

The training concepts are similar. Tensorflow releases vgg-16's architecture in tf-slim's library. <br />
Once we define which parts of layers to be loaded and wihch to be trained, we can load the pretrained vgg.16 checkpoint file to initialize. <br />
The trained layers are the last 3 layers fully-connected 6, fc7 and fc8. There are Dropout layers between each of them. <br />
In this case **GradientDescent** optimizer performs very well. <br />
Tensorflow also offer another preprocessing library to do data augmentation.<br />
The loss function is still **softmax cross entropy**.<br />
The streaming accuracy rises and the loss drops stably. <br />
With batch size 32 and 80 epochs, the model roughly reach convergence.<br />

When it reaches convergence, the training accuracy is 90% and the testing accuracy is .<br />

However, the performance is surely can be better.<br />
Due to I've spent most of my time on Inception-Resnet-v2 experiment, I have got quite less time to tune vgg-16 model.<br />
I haven't tried many combinations of paramaters, so I believe that the current set of paramaters is not the best state of this model.<br />


## Installation
Required package: Tensorflow, tf.slim, numpy and anything the code in trainer file imports.<br />
The pretrained models can be found [here](https://arxiv.org/abs/1512.01881).<br />
To run the trainer code, the frames and labels must be in following sturctures:<br />
frames\<br />
labels\<br />
homework1\<br />
	vgg_16_trainer.py<br />
	inception_resnet_trainer.py<br />
To run the code just simply type **python [whatever].py**.<br />

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


