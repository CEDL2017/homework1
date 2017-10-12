# 楊皓崴 <span style="color:red">(0656706)</span>

#Project 1: Deep Classification

## Overview
The project is related to 
> [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)


## Implementation
1. 前處理
圖檔過大容易耗盡deepQ資源，故將長寬縮小24倍(45*80)  
整理成一般分類問題資料儲存之  

```python
    file = os.path.join(path4,"Image"+str(fileN)+".png")
    pngfile = Image.open(file)
    pngfile.thumbnail(size, Image.ANTIALIAS)
    temp = np.asarray( pngfile, dtype="uint8" )
``` 

2. 模型結構
    * conv layers (3*3 filter) *3
    * fully connect *2

```python
	def alex_net(_X, _weights, _biases, _dropout):
    	_X = tf.reshape(_X, shape=[-1, 45, 80, 3])

    	conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    	pool1 = max_pool('pool1', conv1, k=2)
    	norm1 = norm('norm1', pool1, lsize=4)
    	norm1 = tf.nn.dropout(norm1, _dropout)

	    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    	pool2 = max_pool('pool2', conv2, k=2)
    	norm2 = norm('norm2', pool2, lsize=4)
    	norm2 = tf.nn.dropout(norm2, _dropout)

    	conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    	pool3 = max_pool('pool3', conv3, k=2)
    	norm3 = norm('norm3', pool3, lsize=4)
    	norm3 = tf.nn.dropout(norm3, _dropout)

    	dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    	dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')

    	dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')

    	out = tf.matmul(dense2, _weights['out']) + _biases['out']
    	return out
``` 

## Installation
* Required packages.
    * tensorflow
    * numpy
* How to compile from source?  
    * python preprocessing.py
    * python process01.py / python process02.py

### Results

<table border=1>
<tr>
<td>
process01.py
</td>
<td>
process02.py
</td>
</tr>

<tr>
<td>
<img src="https://github.com/w95wayne10/homework1/blob/master/results/output01.PNG" width="80%"/>
</td>
<td>
<img src="https://github.com/w95wayne10/homework1/blob/master/results/output02.PNG"  width="80%"/>
</td>
</tr>

</table>

嘗試增加conv層數，但沒有效果，猜測圖片縮小過多所致，但操作時間過久來不及嘗試其他尺寸..
