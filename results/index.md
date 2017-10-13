# Your Name <span style="color:red">(id)</span>

#Project 1: Deep Classification

## Overview
The project is related to finetuning VGG16. 
Reference 
https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c
https://github.com/bgshih/vgg16.tf

## Implementation
1. One load in data 
2. Two
	load in pretrained VGG16
	restore VGG16 weight
	delete FC8 layer cause for our own classification task.
3. Three
	finetune from pretrained VGG16
	
VGG architecture 
![](http://book.paddlepaddle.org/03.image_classification/image/vgg16.png)
original VGG paper : https://arxiv.org/pdf/1409.1556.pdf

finetune 

## Installation
Tensorflow, numpy, scipy...etc

### Results
test accuracy : 0.48 (I will train a pretrain resnet50 to surpass 0.6)




