# 賴承薰 <span style="color:red">(105061583)</span>

## HW 1: Deep Classification

## Overview
The project is related to 
> handcam object classification


## Implementation
1. Deep-learning based training model


* ResNet based network with 16 layers

```
net = slim.conv2d(inputs, 64 , [7, 7], stride = 2, scope = 'conv1')
net = slim.max_pool2d(net, kernel_size = [3, 3], stride = 2, padding = 'SAME', scope = 'max_pool1')
short_cut = net

net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope = 'conv2_1')
net = tf.add(net, short_cut)
short_cut = net
net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope = 'conv2_2')
net = tf.add(net, short_cut)
short_cut = net
net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope = 'conv2_3')
net = tf.add(net, short_cut)

net = slim.conv2d(net, 128, [3, 3], stride = 2, scope = 'conv3_1')
net = slim.conv2d(net, 128, [3, 3])
short_cut = net
net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope = 'conv3_2')
net = tf.add(net, short_cut)
short_cut = net
net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope = 'conv3_3')
net = tf.add(net, short_cut)
short_cut = net
net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope = 'conv3_4')
net = tf.add(net, short_cut)

net = slim.conv2d(net, 256, [3, 3], stride = 2, scope = 'conv4_1')
net = slim.conv2d(net, 256, [3, 3])
short_cut = net
net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope = 'conv4_2')
net = tf.add(net, short_cut)
short_cut = net
net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope = 'conv4_3')
net = tf.add(net, short_cut)
short_cut = net
net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope = 'conv4_4')
net = tf.add(net, short_cut)
		
net = slim.avg_pool2d(net, kernel_size = [3, 3], padding = 'SAME')
net = slim.flatten(net)
logits = slim.fully_connected(inputs = net, num_outputs = NUM_CLASS, scope = 'fc')
```

* Cross Entropy with logits

* Adam Optimizer

```
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_predict, labels= y_label))
optimizer = tf.train.AdamOptimizer(LR).minimize(loss)
```

## Installation
* Tensorflow
    
    * contrib.slim
    
    > construct the architecture

* Numpy

* PIL Image

    > handle the image problem

* os

    > help reading files


## Results
#### Validation accuracy = 0.547222266667
#### Validation loss     = 2.25264353333
<table border=1>
<tr>
<td>
<img src="螢幕快照 2017-10-11 下午8.11.21.png" width="50%"/>
</td>
</tr>

<tr>
<td>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

</table>


