# 林怡君 <span style="color:red">(106062514)</span>

#Homework 1: Deep Classification

## Overview
The project is related to 
> Classifying the object categories by implementing Inception-ResNet-v2 network.


## Implementation
1. Data_Generator.py
	* Since the training data and testing data are all mixed in folders, I came up with an idea to store the file paths in the text files respectively. 
 	* Running Data_Generator.py to divide all the file paths into four files.The four files are ‘train_image.txt’, ‘train_label.txt’, ‘test_image.txt’ and ‘test_label.txt’.
2. train.py
    * Load data. Before training, the pre-processing is needed. After loading data from the four text files, I used vgg_preprocessing.py to pre-process the data. Though inception_preprocessing.py is recommended, vgg_preprocessing.py spends less time.
    * Construct the whole network. I chose the Inception-ResNet-v2 as the pre-trained model since the accuracy seems to the highest.
	* Fine-tune the network.

## Installation
* Tensorflow
* Tensorflow.contrib.slim
* vgg_preprocessing.py
* inception_resnet_v2.py
* numpy
* os

### Results

<table border=1>
<tr>

</tr>

<tr>
<img src="result.png width="24%" alt = "results" style = "float:middle;" />
</tr>

</table>


