# 劉豐源 <span style="color:red">(106062519)</span>

#Project 1: Deep Classification

## Overview
The project is related to 
> 練習使用CNN進行影像辨識   <br />
> 標準(benchmark): AlexNet > 60%

## Implementation
1. (92%) AlexNet
	* 將input data轉成 ndarray(227,227,3)
	* 右手的資料，將其左右翻轉 (思路:讓右手的data跟左手的data角度一致化)
	* 使用AlexNet進行訓練
	* 訓練過程使用的資料有做shuffle (訓練資料分布較為均勻)
	* (227,227,3)的資料略大，於是分批讀取、訓練
	* Reference: http://blog.csdn.net/wmy199216/article/details/71171401
2. (53%) CNN_basic
	* 原本是用來預測cifar10的model (71%原精準度)
	* 容易訓練、易懂好寫
	* 是我一開始作業練習使用的model
	* Reference: (書本) TensorFlow+Keras深度學習人工智慧實務應用 
	* Reference: http://tensorflowkeras.blogspot.tw/2017/10/kerasdeep-learningcnncifar-10.html
3. (50%) Mnist
	* 原本是用來預測手寫數字的model (99%原精準度)
	* 使用灰階圖進行訓練
	* 雖然灰階圖流失了許多資訊，但仍然保有50%的精確度
	* Reference: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

```
Code highlights
```

## Installation
* Other required packages.
from PIL import Image #read image data
from scipy import misc #read image data
import numpy as np #ndarray  
import os #get all file name in the folder



* How to compile from source?
  Tensorflow, Keras

### Results

<table border=1>
<tr>
<td>
[9,8]  <br />
train data: images: (1144, 227, 227, 3)  labels: (1144,)   <br />
test  data: images: (838, 227, 227, 3)  labels: (838,)    <br />
Train on 915 samples, validate on 229 samples     <br />
Epoch 1/1       <br />
915/915 [====================] - 1s - loss: 0.1049 - acc: 0.9683 - val_loss: 0.4572 - val_acc: 0.9083 <br />
<br />
[9,4] <br />
train data: images: (1178, 227, 227, 3)  labels: (1178,) <br />
test  data: images: (1316, 227, 227, 3)  labels: (1316,) <br />
Train on 942 samples, validate on 236 samples <br />
Epoch 1/1 <br />
942/942 [====================] - 1s - loss: 0.1442 - acc: 0.9459 - val_loss: 0.3446 - val_acc: 0.9110 <br />
<br />
[9,5]  <br />
train data: images: (1460, 227, 227, 3)  labels: (1460,)  <br />
test  data: images: (934, 227, 227, 3)  labels: (934,)  <br />
Train on 1168 samples, validate on 292 samples  <br />
Epoch 1/1  <br />
1168/1168 [====================] - 1s - loss: 0.0625 - acc: 0.9795 - val_loss: 0.1606 - val_acc: 0.9623 <br />



<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>


(Average) Test Accuracy= 0.918689968214 <br />
(Average) Test Loss= 0.275209411445 <br />

<tr>
<td>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

</table>


