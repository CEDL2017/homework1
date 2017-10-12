# Your Name <span style="color:red">(id)</span>
name: 何元通    ID: 105062575  

# Homework 1 : Deep Classification

## Overview
Recently, the technological advance of wearable devices has led to significant interests in recognizing human behaviors in daily life (i.e., uninstrumented environment). Among many devices, egocentric camera systems have drawn significant attention, since the camera is aligned with the field-of-view of wearer, it naturally captures what a person sees. These systems have shown great potential in recognizing daily activities(e.g., making meals, watching TV, etc.), estimating hand poses, generating howto videos, etc.  
  
Despite many advantages of egocentric camera systems, there exists two main issues which are much less discussed. Firstly, hand localization is not solved especially for passive camera systems. Even for active camera systems like Kinect, hand localization is challenging when two hands are interacting or a hand is interacting with an object. Secondly, the limited field-of-view of an egocentric camera implies that hands will inevitably move outside the images sometimes.  
  
HandCam, a novel wearable camera capturing activities of hands, for recognizing human behaviors. HandCam has two main advantages over egocentric systems : (1) it avoids the need to detect hands and manipulation regions; (2) it observes the activities of hands almost at all time.  
  
For this homework, we are asked to solve an image classification problem. What we should do is to implement a deep-learning-based approach to determine what things the user takes in hand. 


## Implementation
1. The first step of my algorithm is to perform image pre-process. I load the image and resize it to a specific size. Then, save it as a tfrecord file. The file is used for queue loading process in tensorflow. After these pre-process, I load the data and start to training or evaluation. Once the training finished, the parameters of the model will be save in log directory. Also, once we want to evaluation, the model will be loaded in as well.  
  
2. For the file-loading approach, I consider that the normal way to load in all the images will not perform well. For the first reason that the data is in a large size and this may cause a memory insufficient problem. Second, I think that there must be a build-in file-loading method in tensorflow to efficiently load the file. Thus, I found `tfrecord`, which is one of the efficient file-loading approach in tensorflow. It will save the data in a binary file. Then, load in the files with the queue. This can make the data more convenient on storage, copying and movement. Furthermore, since the data will be loaded from the queue, this help as save more memory compared with the fasion that load all images first. Comapared with loading all first, it performs much sufficient and faster, though it is much complicated on implementation and pre-process for the `tfrecord` specific file.  
  
3. In this assignment, what we have to do is image classification. Considering image classification has been a well-performed field in computer vision. I try to use VGG-16, which is one of the classic convolutional neural network in classfication, as my network structure. The code of my model is shown as below.  

      I do not complete follow the VGG-16 and use its pre-train model. I modify the channel size of each layer. I consider that the modification can make it run faster as the resource and time limit.

```
        self.conv_1_1 = self.conv_layer(image, [3, 3, 3, 4], "conv1_1")
        self.relu_1_1 = tf.nn.relu(self.conv_1_1)
        self.conv_1_2 = self.conv_layer(self.relu_1_1, [3, 3, 4, 4], "conv1_2")
        self.relu_1_2 = tf.nn.relu(self.conv_1_2)
        self.pool1 = self.max_pool_2_2(self.relu_1_2)

        # conv2
        self.conv_2_1 = self.conv_layer(self.pool1, [3, 3, 4, 8], "conv2_1")
        self.relu_2_1 = tf.nn.relu(self.conv_2_1)
        self.conv_2_2 = self.conv_layer(self.relu_2_1, [3, 3, 8, 8], "conv2_2")
        self.relu_2_2 = tf.nn.relu(self.conv_2_2)
        self.pool2 = self.max_pool_2_2(self.relu_2_2)

        # conv3
        self.conv_3_1 = self.conv_layer(self.pool2, [3, 3, 8, 16], "conv3_1")
        self.relu_3_1 = tf.nn.relu(self.conv_3_1)
        self.conv_3_2 = self.conv_layer(self.relu_3_1, [3, 3, 16, 16], "conv3_2")
        self.relu_3_2 = tf.nn.relu(self.conv_3_2)
        self.conv_3_3 = self.conv_layer(self.relu_3_2, [3, 3, 16, 16], "conv3_3")
        self.relu_3_3 = tf.nn.relu(self.conv_3_3)
        self.pool3 = self.max_pool_2_2(self.relu_3_3)

        # conv4
        self.conv_4_1 = self.conv_layer(self.pool3, [3, 3, 16, 32], "conv4_1")
        self.relu_4_1 = tf.nn.relu(self.conv_4_1)
        self.conv_4_2 = self.conv_layer(self.relu_4_1, [3, 3, 32, 32], "conv4_2")
        self.relu_4_2 = tf.nn.relu(self.conv_4_2)
        self.conv_4_3 = self.conv_layer(self.relu_4_2, [3, 3, 32, 32], "conv4_3")
        self.relu_4_3 = tf.nn.relu(self.conv_4_3)
        self.pool4 = self.max_pool_2_2(self.relu_4_3)

        # conv5
        self.conv_5_1 = self.conv_layer(self.pool4, [3, 3, 32, 64], "conv5_1")
        self.relu_5_1 = tf.nn.relu(self.conv_5_1)
        self.conv_5_2 = self.conv_layer(self.relu_5_1, [3, 3, 64, 64], "conv5_2")
        self.relu_5_2 = tf.nn.relu(self.conv_5_2)
        self.conv_5_3 = self.conv_layer(self.relu_5_2, [3, 3, 64, 64], "conv5_3")
        self.relu_5_3 = tf.nn.relu(self.conv_5_3)
        self.pool5 = self.max_pool_2_2(self.relu_5_3)

        # fc6
        self.fc6 = self.fc_layer(self.pool5, 512, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)

        # fc7
        self.fc7 = self.fc_layer(self.relu6, 1024, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        # fc8
        self.fc8 = self.fc_layer(self.relu7, OUTPUT_SIZE, "fc8")
        self.prediction = tf.nn.softmax(self.fc8)

```


## Installation
1. Required Packages:
The code is programed under ubuntu environment with latest version of python3.  
For deep learning, tensorflow are used in my code.  
The other python packages required are as shown below:  
   * numpy
   * opencv2
2. Command to run the codes:
   * If you want to train the network and this is first time you run the codes:
```
python3 Main.py --train --pre
```
   It is notice that you just need `--pre` command for the first time running. The command ask the program to produce tfrecord file, which is used for the pre-process and queue loading for input images.  
   Also, `--train` is optional in training procedure. The program runs training procedure as default. And, the command tells the program we are training now.  
  
   * If you want to test the network:
```
python3 Main.py --test --pre
```
   It works the same functionality as training procedure of `--pre`. And, `--test` works similar to `--train`. Nonetheless, noticed that this is necessary and can not be ignored in testing procedure.

### Results

Some settings are shown as following:
    * loss: softmax cross entropy
    * optimizer: adamoptimizer
    * learning rate = 1e-5
    * batch size = 16
    * epoch = 10
      
For the loss and optimizer, I have used cross entropy and gradient descent. But, with the suggestion from an experienced classmate, I modify it to the current loss function and optimizer as both of the two is more appropriate to a classification work. Also, it raises the precision approximate 2%.

The learning rate is related to the optimizer. It is set to 1e-3 as default. I have tried 1e-1 to 1e-8, and 1e-4 and 1e-5 performed better results. Eventually, I randomly choose one of them as their results and convergence speed is close.

As to epoch, I have set it larger. Nonetheless, the performance seems not change much. Consequently, I set it to 10 for quickly implementation.
    
Finall precision: 50.0548245614%


