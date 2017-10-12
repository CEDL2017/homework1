# 徐慧文 <span>106065507</span>

#Homework1: Deep Classification

## Overview
The homework is related to classify objects by hand camera. I used  VGG19 network to implement the homework.
Why do I choose VGG19? I think that not only the network is better than alexNet but also the neural network is complete and applicable in dataset at this time.  
My environment:
* Language:  python
* OS : unbuntu
* cnn: VGG19
## Implementation
1. **One**(How to implement?)
	* Read the images and labels to lists.
	* Transform the data type into the TFrecord type that can accelerate training time.
	* It needs to decode the tfrecord file before producing image_batch and label_batch.
	* Program starts to train after throwing Image_Batch to the VGG19 model.
	* When training and calculating training accuracy are finished, we can throw test data into model and then get accuracy of test data. 
2. **Two**( code hightlight)

Each convolution layer's output parameters are not the same as the VGG paper because our classes are just 24 :
```
x_feature = conv('conv1_1', images, 8, kernel_size=[3,3], stride=[1,1,1,1])   
x_feature = conv('conv1_2', x_feature, 8, kernel_size=[3,3], stride=[1,1,1,1])
x_feature = Max_pool('pool1', x_feature, kernel=[1,2,2,1], stride=[1,2,2,1])

x_feature = conv('conv2_1', x_feature, 16, kernel_size=[3,3], stride=[1,1,1,1])    
x_feature = conv('conv2_2', x_feature, 16, kernel_size=[3,3], stride=[1,1,1,1])
x_feature = Max_pool('pool2', x_feature, kernel=[1,2,2,1], stride=[1,2,2,1])
```
Using  Nadamoptimizer that combines Nesterov and adam's advantage:  
```
optimizer = tf.contrib.opt.NadamOptimizer().minimize(loss)
```
Transform data into tfrecord type:
```
 writer = tf.python_io.TFRecordWriter(filename)
 for i in np.arange(0, n_samples):
    image = Image.open(images[i]) # type(image) must be array!
    image = image.resize((227,227)) 
    image_raw = image.tobytes()
    label = int(labels[i])
    example = tf.train.Example(features = tf.train.Features(feature = {
                                        'label':feature_int64(label),
                                        'image_raw': feature_bytes(image_raw)}))
    writer.write(example.SerializeToString())
```
## Installation
* **Required packages**  
   * tensorflow 
   * numpy
   * scipy
   * scikit-image
  
	
* **How to compile from source?**
  1. Clone my code files to local.  
  2. Change the train path and data path in readfile.py.  
     (**Just modify some path part before place_name list**)
	 
     **obj_label_path、train_path**
	 
	 ```python
     (readfile.py)
     def read_file():
     	place_name = { 0:"house", 1:"lab", 2:"office" }
     	if is_leftHand : #左手
            
            obj_label_path = '/home/viplab/Downloads/labels/'+place_name[key]+'/obj_left'+str(j)+'.npy'
            train_path = '/home/viplab/Downloads/frames/train/'+place_name[key]+'/'+str(j)+'/Lhand/'
        else :  #右手    
            obj_label_path = '/home/viplab/Downloads/labels/'+place_name[key]+'/obj_right'+str(j)+'.npy'
            train_path = '/home/viplab/Downloads/frames/train/'+place_name[key]+'/'+str(j)+'/Rhand/'
     ==================================================================================
	 def read_test():
     	place_name = { 0:"house", 1:"lab", 2:"office" }
     	if is_leftHand : #左手
            
            obj_label_path = '/home/viplab/Downloads/labels/'+place_name[key]+'/obj_left'+str(index[key][j])+'.npy'
            train_path = '/home/viplab/Downloads/frames/test/'+place_name[key]+'/'+str(j)+'/Lhand/'

        else :  #右手
            obj_label_path = '/home/viplab/Downloads/labels/'+place_name[key]+'/obj_right'+str(index[key][j])+'.npy'
            train_path = '/home/viplab/Downloads/frames/test/'+place_name[key]+'/'+str(j)+'/Rhand/'
			
     ```
	 
  3. Modify train_log_dir 、tfrecords_file and save_dir of 'run_training()' function in trainS.py
     Modify log_dir path、tfrecords_file path、save_dir path of 'evaluate()' function
     
    ```python
    (trainS.py)
    def run_training():
  		train_log_dir = '/home/viplab/Desktop/vgg_var2/log/train/'  
    	tfrecords_file ='/home/viplab/Desktop/vgg_var2/tfrecord/train.tfrecords'#記得最後要是train.tfrecords
    	save_dir = '/home/viplab/Desktop/vgg_var2/tfrecord/'
  ===================================================================================
  	def evaluate():
    	log_dir = '/home/viplab/Desktop/vgg_var2/log/train/' #跟run_training()裡的train_log_dir一樣路徑
        tfrecords_file ='/home/viplab/Desktop/vgg_var2/tfrecord/test/test.tfrecords'#記得最後要是test.tfrecords
        save_dir = '/home/viplab/Desktop/vgg_var2/tfrecord/test/'
    ```
	4. python trainS.py , it will show option in console. You can insert number that you want to do.
  ```
  >>python trainS.py
  ```
 

### Results

|   Accuracy   |VGG19_1 |VGG19-2|
| :----:       | :----:|:----:|
|    train(%)  |  50    |  62.5 |
|     test(%)  |  50.09 |  50.25|

VGG19_1 : learning-rate =  0.01 ; Epoch = 50 ; optimizer: Adam  
VGG19_2 : learning-rate =  0.01 ; Epoch = 50 ; optimizer: nadam


  VGG19_1 and VGG19_2 are different from  optimizer's method. VGG19_1 used adam-optimizer and VGG19_2 used nadam-optimizer. Two optimizers can continuously revise learning-rate by themself when traing is processing. Just nadam is more stable than adam, but Their results are not clear difference.


My training accuracy cannot be improving. I tried many methods such as learning-rate modification、optimizer、epoch number increase and so on, but result is still bad.   
I thinked that maybe I don't use head data to reinforce network or training network needs to be revised .
     