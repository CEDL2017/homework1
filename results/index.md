# 陳則銘 <span style="color:red">105062576</span>

#Project 5: Deep Classification

## Overview
Recently, the technological advance of wearable devices has led to significant interests in recognizing human behaviors in daily life (i.e., uninstrumented environment). Among many devices, egocentric camera systems have drawn significant attention, since the camera is aligned with the field-of-view of wearer, it naturally captures what a person sees. These systems have shown great potential in recognizing daily activities(e.g., making meals, watching TV, etc.), estimating hand poses, generating howto videos, etc.

Despite many advantages of egocentric camera systems, there exists two main issues which are much less discussed. Firstly, hand localization is not solved especially for passive camera systems. Even for active camera systems like Kinect, hand localization is challenging when two hands are interacting or a hand is interacting with an object. Secondly, the limited field-of-view of an egocentric camera implies that hands will inevitably move outside the images sometimes.

HandCam (Fig. 1), a novel wearable camera capturing activities of hands, for recognizing human behaviors. HandCam has two main advantages over egocentric systems : (1) it avoids the need to detect hands and manipulation regions; (2) it observes the activities of hands almost at all time.



## Implementation
1. Using a dataset recorded by hand camera system.
	* The dataset consists of 20 sets of video sequences (i.e., each set includes two HandCams synchronized videos) captured in three scenes: a small office, a mid-size lab, and a large home.)
	* We want to classify one kind of hand state: object categories. At the same time, a synchronized video has two sequence need to be labeled, the left hand states and right hand states.
	* For the classification task (i.e. object categories), there are forty sequences of data. We split the dataset into two parts, half for training, half for testing. The object instance is totally separated into training and testing.
	* Details of object categories (24 classes, including free)
	```
	Obj = { 'free':0,
        'computer':1,
        'cellphone':2,
        'coin':3,
        'ruler':4,
        'thermos-bottle':5,
        'whiteboard-pen':6,
        'whiteboard-eraser':7,
        'pen':8,
        'cup':9,
        'remote-control-TV':10,
        'remote-control-AC':11,
        'switch':12,
        'windows':13,
        'fridge':14,
        'cupboard':15,
        'water-tap':16,
        'toy':17,
        'kettle':18,
        'bottle':19,
        'cookie':20,
        'book':21,
        'magnet':22,
        'lamp-switch':23}
	```
	* Because of the complex structures of the input data folder, I use os.walk and fnmatch to read my file:
	```
	import os
	import fnmatch
	if is_train:
            _dir = 'frames/train'
            num1 = '[1234]'
            num2 = '[123]'
        else:
            _dir = 'frames/test'
            num1 = '[5678]'
            num2 = '[456]'
	
	## For reading images ##
        for root, dirs, files in os.walk(_dir):
            files.sort(key=lambda x:(x[5:-4]))
            for f in files:
                image_list.append(os.path.join(root,f))
                
	## For reading labels ##
        for root, dirs, files in os.walk('labels/'):
            files.sort()
            seq = ''
            if root == 'labels/lab' :
                seq = 'obj_*'+num1+'.npy'
            else:
                seq = 'obj_*'+num2+'.npy'
            for f in fnmatch.filter(files, seq):   
                hand = np.load(os.path.join(root,f))
                label_list += [int(i) for i in hand]
	```
	
2. Using VGG16 as deep-learning-based method
	```
	x = tools.conv('conv1_1', x, 8, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    	x = tools.conv('conv1_2', x, 8, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

	x = tools.conv('conv2_1', x, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.conv('conv2_2', x, 16, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

	x = tools.conv('conv3_1', x, 32, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.conv('conv3_2', x, 32, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.conv('conv3_3', x, 32, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

	x = tools.conv('conv4_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.conv('conv4_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.conv('conv4_3', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

	x = tools.conv('conv5_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.conv('conv5_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.conv('conv5_3', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
	x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            

	x = tools.FC_layer('fc6', x, out_nodes=512)
	x = tf.nn.dropout(x, 0.5)
	#x = tools.batch_norm(x)
	x = tools.FC_layer('fc7', x, out_nodes=512)
	x = tf.nn.dropout(x, 0.5)
	#x = tools.batch_norm(x)
	x = tools.FC_layer('fc8', x, out_nodes=n_classes)
	```
	* adding dropout after fc6 and fc7 layer for better results
3. For data images, I'll resize them by (weight, height) = (224, 224) as my input images.
4. Using softmax for loss funtion
	```
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
	```
5. Optimized by AdamOptimizer, GradientDescent
	``` Adam
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	```
	```GradientDescent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
	```
6. learning rate set as 0.01, 0.1, 0.001

Reference to
> https://github.com/kevin28520/My-TensorFlow-tutorials

## Installation
* Install python3 and tensorflow first
* Set the dataset directory in input_data.py , data_dir='put/your/dataset/dir'
* In train.py:
```
## For training, uncomment #train()  
#train()   
## For testing, uncomment #evaluate() 
#evaluate()
```

### Results

* optimizer: AdamOptimizer

| Learning Rate | Testing Accurancy |
| --- | --- |
| 0.1 | 50.96% |
| 0.001 | 50.33% |

* optimizer: GradientDescentOptimizer

| Learning Rate | Testing Accurancy |
| --- | --- |
| 0.1 | 50.62% |
| 0.001 | 50.09% |

