# 邱浩翰 <span style="color:red">(105061607)</span>

#Project 5: Deep Classification

## Overview

<center>
<img src="README_files/overview.png" alt="overview" style="float:middle;">
</center>

The project is related to 
* handcam object classification
* VGG16 
* Reference to:

code
>https://github.com/kevin28520/My-TensorFlow-tutorials

vgg16
>https://arxiv.org/abs/1409.1556
## Implementation
1. Load data
	* image & label
	```
	for dirPath, dirNames, fileNames in os.walk(label_path):
    		for i in range(len(train_label_file)):
        		train_labels = np.hstack((train_labels, np.load(label_path + train_label_file[i], mmap_mode='r')))

	for dirPath, dirNames, fileNames in os.walk(label_path):
    		for i in range(len(test_label_file)):
        		test_labels = np.hstack((test_labels, np.load(label_path + test_label_file[i], mmap_mode='r')))

	for i in range(len(train_image_file)):
    		for dirPath, dirNames, fileNames in os.walk(train_image_path + train_image_file[i]):
        		for f in fileNames:
            			train_images.append(os.path.join(dirPath, f))

	for i in range(len(test_image_file)):
    		for dirPath, dirNames, fileNames in os.walk(test_image_path + test_image_file[i]):
        		for f in fileNames:
            			test_images.append(os.path.join(dirPath, f))
	```
	
	
2. Training
	* Optimizer
	```
	loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_logits, labels= y_label))#tools.loss(y_logits, y_label) 
	optimizer = tf.train.AdamOptimizer(LR).minimize(loss_func)
	
	```

3. Testing
	* Accuracy_evaluation
	```
	correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y_label, 1))
	accuracy = tf.cast(tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32)), dtype=tf.float32)#tools.accuracy(y_logits, y_label)

	```

4. Architecture
	* VGG16
	```
	def VGG16N(x, n_classes, is_pretrain=True):
    
		with tf.name_scope('VGG16'):

			x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)   
			x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			with tf.name_scope('pool1'):    
			    x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

			x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)    
			x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			with tf.name_scope('pool2'):    
			    x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)



			x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			with tf.name_scope('pool3'):
			    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)


			x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			with tf.name_scope('pool4'):
			    x = tools.pool('pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)


			x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
			with tf.name_scope('pool5'):
			    x = tools.pool('pool5', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            


			x = tools.FC_layer('fc6', x, out_nodes=4096)        
			#with tf.name_scope('batch_norm1'):
			    #x = tools.batch_norm(x)           
			x = tools.FC_layer('fc7', x, out_nodes=4096)        
			#with tf.name_scope('batch_norm2'):
			    #x = tools.batch_norm(x)            
			x = tools.FC_layer('fc8', x, out_nodes=n_classes)

			return x
	```

6. Using pretrained weights or not
	```
	pre_trained_weights = 'vgg16.npy'
	tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])
	```

7. Parameters
	* BATCH_SIZE = 12
	* INPUT_WIDTH = 224
	* INPUT_HEIGHT = 224
	* EPOCH = 50
	* LR = 10**(-4) # learning rate
	* NUM_CLASS = 24

## Installation
* import VGG and tools (for network)
* Set the dataset directory 
* switch train or test with comment
	```
	#for test : comment line261~321
	'''
	# strat trainng
	startTime = time()
	init = tf.global_variables_initializer()


	with tf.Session() as sess:
    		sess.run(init)
		.
		.
		.
	# duration calculating
	duration = time()-startTime
	print('duration = ', duration)
	#### End of training ####	
	'''
	```
	
### Results

| Learning Rate |Loss| Testing Accurancy |
| --- | --- | --- |
| 0.001 | 2.67885 | 52.61% |



* Training time

> about 12 hours with GTX 1080 Ti
