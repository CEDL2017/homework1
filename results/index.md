# 許菀庭 <span style="color:red">(105061525)</span>

# Homework 1: Deep Classification

## Overview
The goal of this project is to classify the objects of the HandCam images.


## Implementation

1. My model is implemented in Pytorch.

2. Model:

	* I choose **Resnet18** and **Resnet50** as the image classification model.
	* I use the model **pretrained on ImageNet dataset** to initialize the model parameters except the fully-connected layer (output layer).
	* Since the number of classes is 24 for HandCam dataset, I change the output size of the fully-connected layer to 24.

3. Data Preprocessed and Data Augmentation:
	
	* All images are scaled to 256 * 256 and then cropped to 224 * 224.
	* All images are normalized using the mean and variance of ImageNet dataset.
	* When training, use **random crop** and **random horizontal flip** to do data augmentation.
	* When testing, use center crop.
	<br>

	```python
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	if split == 'train':
	    self.transform = transforms.Compose([transforms.Scale(256),
	                                         transforms.RandomSizedCrop(224),
	                                         transforms.RandomHorizontalFlip(),
	                                         transforms.ToTensor(),
	                                         normalize])
	else:
	    self.transform = transforms.Compose([transforms.Scale(256),
	                                         transforms.CenterCrop(224),
	                                         transforms.ToTensor(),
	                                         normalize])
	```

## Installation
* Required Pytorch.
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


