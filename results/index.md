# 陳聖諺 <span style="color:red">(105061604)</span>

#Homework1: Deep Classification

## Overview
The project is related to 
> 1. Deep-learning-based method
  2. Alexnet
  3. Classification

## Implementation
[Load Input] 2-way
1. Using os.listdir  (need to sort)    

code example:

    for im in os.listdir(pathh1):
        
        test_path = os.path.join(pathh1, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

      test_path_listr.extend(test_path_list)
      test_path_list=[]
2. Using range (read images in order but need to know exactly how many data in a file)

code example:

    for i in range(572):
        i=i+1
        img_path = '../Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

[Difference between loading Image and Label]

When loading image, we have to load the path first. If we load all the images at the same time, the RAM will be overloaded and the computer will be crashed. 
Therefore, we will actually read the image until we are doing the training process. 
However, since the labels are not quite a big file, we can load them all in the beginning.

[Shuffle the Data-Keep the image and label matched]

#打亂順序
num_example=data_list.shape[0]    #how many images in the list
#print(num_example)               #check the number of images (match with labels')
arr=np.arange(num_example)        #create a number list
np.random.shuffle(arr)            #shuffle the number lsit
data_list=data_list[arr]          #make the data_list into a new order
#print(data_list)
label=label[arr]                  #make the label into a new order
#print(label.shape[0])

[minibatch]

We will train the classifier in a batch, so in this case the images have to be chosen batch by batch without losing the identity of itselves.
We will save the index of certain batch of images and labels in variable 'excerpt' and output two vectors
:input[excerpt] and targets[excerpt]
which input is the path of the batch images and targets is the corresponding label information

code example:
    
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
                
        yield inputs[excerpt], targets[excerpt]
[Training]
Before feeding the training images into dict, we have to read and deal with the images first.
This part requires several processing.

1. imread 
2. resize
3. normalization
4. reshape
5. append

code example:
        
	temp =[]
        #print(x_train_a)
        for i in range(batch_size):            
        #print(io.imread(x_train_a[1]))  
        
            tmp = io.imread(x_train_a[i])
            #print(i)
            tmp = np.resize(tmp,(w,h,c))
           # print(tmp)
            tmp = tmp/256
            #tmp = np.reshape(tmp,[-1,w,h,c])
            #print(tmp.shape)
            temp.append(tmp)
	    
[Deep Structure]

I tried to fix Alexnet in a little extent.
1.number of layers (from 4~8)
2.modify the 3fc layer into 1fc layer
3.Different input image size (100*100 ~ 400*400)
4.Different batchsize (1~48)
(etc)

However the accuracy will eventually stop by around 50%.
Lose cannot be lower than 2

## Installation

	from skimage import io,transform
	import glob
	import os
	import re
	import numpy as np
	import time
	import tensorflow as tf


### Results

<table border=1>
<tr>
<td>
<img src=".jpg" width="24%"/>
<img src="/home/benita/Downloads/CEDLHW1.png"  width="24%"/>
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


