# 劉豐源 <span style="color:red">(106062519)</span>

#Project 1: Deep Classification

## Overview
The project is related to 
> 練習使用CNN進行影像辨識   <br />
> 標準(benchmark): AlexNet > 60%

## Implementation
1. (54%) AlexNet
	* 將input data轉成 ndarray(227,227,3)
	* 右手的資料，將其左右翻轉 (思路:讓右手的data跟左手的data角度一致化)
	* 使用AlexNet進行訓練
	* 訓練過程使用的資料有做shuffle (訓練資料分布較為均勻)
	* <a href="https://github.com/smileyung/homework1/blob/master/code/(54%25)%20AlexNet.ipynb_v2.0.ipynb/" title="Code_AlexNet Search">Code_AlexNet</a>
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

### Code highlights (AlexNet construct)
```
model = Sequential()  
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(24,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()
```

## Installation (packages)
* Environment   <br />
  Keras 2.0.8  <br />
  Tensorflow 1.3.0  <br />
  conda 4.3.25  <br />

* Read file  <br />
#read image data  <br />
from PIL import Image    <br />
from scipy import misc    <br />
#get all file name in the folder  <br />
import os    <br />

* Data preprocess  <br />
#ndarray  <br />
import numpy as np       <br />
#shuffle training data  <br />
from sklearn.utils import shuffle  <br />
#OneHot Encoding      <br />
from keras.utils.np_utils import to_categorical  <br />

* Construct model  <br />
from keras.models import Sequential    <br />
from keras.layers import Dense, Dropout, Activation, Flatten    <br />
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D    <br />


* Graph plot <br />
import matplotlib.pyplot as plt  <br />


### Results

<table border=1>
<tr>
<td>
	
![image](https://github.com/smileyung/homework1/blob/master/results/img/AlexAcc.png)
	
![image](https://github.com/smileyung/homework1/blob/master/results/img/AlexLoss.png)

![image](https://github.com/smileyung/homework1/blob/master/results/img/AlexTestAcc.png)	
	
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

accuracy = 54% (in round 94) <br />
Test Accuracy= 0.541553241303 <br />
Test Loss= 3.43139602775

<tr>
<td>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

</table>


