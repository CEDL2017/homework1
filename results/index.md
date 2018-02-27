# Your Name <span style="color:red">104061213 林倢愷</span>

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

### discussion
原本預計使用pretrained model就可以達到還不錯的成績，
但是最後的結果很慘，
所以來分析一下，
大部分純classification的task用pretrained model+finetune都可以達到還不錯的結果，
我結果差我認為有兩個可能。
1. 本身dataset跟imagenet差異太大，之前看過一個分析是目前用imagenet來train的model都有明顯的缺點在小的物體、薄的物體、半透明的物體．．．等，而老師的dataset大部分的物體都很小，因此model辨識不出來，所以導致result很差。
2. finetune不夠遠，我freeze住除了FC8跟最後兩個conv以外的所有層數，根據上一點，應該要finetune更遠，可能一半的network都finetune才可以達到比較好的result，原本finetune不遠的考量是dataset不是很大，所以才決定只finetune最後幾層。




