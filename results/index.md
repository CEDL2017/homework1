# 邱煜淵 <span style="color:red">(105061634)</span>

#Project 5: Deep Classification

## Overview
The project is related to object classification. There are two different case. One is every task has itself model. The other is just one model handle multiple task (Obj, Ged and FA).
> quote


## Implementation
1. One
	* item
	* item
	
2. spotlight code

```
with tf.name_scope('C1_Conv'):
    W1 = weight([3,3,3,36])
    b1 = bias([36])
    Conv1 = conv2d(x_image, W1)+b1
    Conv1_BN = BN(Conv1, 36)
    C1_Conv = tf.nn.relu(Conv1_BN)  
with tf.name_scope('C2_Conv'):
    W2 = weight([3,3,36,36])
    b2 = bias([36])
    Conv2 = conv2d(C1_Conv, W2)+b2
    Conv2_BN = BN(Conv2, 36)
    C2_Conv = tf.nn.relu(Conv2_BN) 
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pooling(C2_Conv)
    
with tf.name_scope('C3_Conv'):
    W3 = weight([3,3,36,64])
    b3 = bias([64])
    Conv3 = conv2d(C1_Pool, W3)+b3
    Conv3_BN = BN(Conv3, 64)
    C3_Conv = tf.nn.relu(Conv3_BN)
C3_concate = tf.concat([C1_Pool, C3_Conv],3)
with tf.name_scope('C4_Conv'):
    W4 = weight([3,3,100,128])
    b4 = bias([128])
    Conv4 = conv2d(C3_concate, W4)+b4
    Conv4_BN = BN(Conv4, 128)
    C4_Conv = tf.nn.relu(Conv4_BN)    
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pooling(C4_Conv)
    
with tf.name_scope('C5_Conv'):
    W8 = weight([3,3,128,128])
    b8 = bias([128])
    Conv5 = conv2d(C2_Pool, W8)+b8
    Conv5_BN = BN(Conv5, 128)
    C5_Conv = tf.nn.relu(Conv5_BN)
C5_concate = tf.concat([C2_Pool, C5_Conv],3)
with tf.name_scope('C6_Conv'):
    W9 = weight([3,3,256,256])
    b9 = bias([256])
    Conv6 = conv2d(C5_concate, W9)+b9
    Conv6_BN = BN(Conv6, 256)
    C6_Conv = tf.nn.relu(Conv6_BN)    
with tf.name_scope('C3_Pool'):
    C3_Pool = max_pooling(C6_Conv)

with tf.name_scope('C7_Conv'):
    W10 = weight([3,3,256,256])
    b10 = bias([256])
    Conv7 = conv2d(C3_Pool, W10)+b10
    Conv7_BN = BN(Conv7, 256)
    C7_Conv = tf.nn.relu(Conv7_BN)
C7_concate = tf.concat([C3_Pool, C7_Conv],3)
with tf.name_scope('C8_Conv'):
    W11 = weight([3,3,512,512])
    b11 = bias([512])
    Conv8 = conv2d(C7_concate, W11)+b11
    Conv8_BN = BN(Conv8, 512)
    C8_Conv = tf.nn.relu(Conv8_BN)    
with tf.name_scope('C4_Pool'):
    C4_Pool = max_pooling(C8_Conv)
```

## Installation
* Other required packages.
	* python2.7
	* tensorflow
	* numpy
	* PIL
* How to compile from source?
	<p>First, you must resize the input image.<p>
	
	>$ python data_resize.py
	
	Than you can chose which task you want to train.
	
	>$ python train.py or $ python train_obj.py
	
	Last, you can test your model.
	
	>$ python test.py
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


