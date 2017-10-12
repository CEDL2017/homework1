# Your Name <span style="color:red">(id)</span>
name: 何元通    ID: 105062575  

# Homework 1 : Deep Classification

## Overview
Recently, the technological advance of wearable devices has led to significant interests in recognizing human behaviors in daily life (i.e., uninstrumented environment). Among many devices, egocentric camera systems have drawn significant attention, since the camera is aligned with the field-of-view of wearer, it naturally captures what a person sees. These systems have shown great potential in recognizing daily activities(e.g., making meals, watching TV, etc.), estimating hand poses, generating howto videos, etc.  
  
Despite many advantages of egocentric camera systems, there exists two main issues which are much less discussed. Firstly, hand localization is not solved especially for passive camera systems. Even for active camera systems like Kinect, hand localization is challenging when two hands are interacting or a hand is interacting with an object. Secondly, the limited field-of-view of an egocentric camera implies that hands will inevitably move outside the images sometimes.  
  
HandCam, a novel wearable camera capturing activities of hands, for recognizing human behaviors. HandCam has two main advantages over egocentric systems : (1) it avoids the need to detect hands and manipulation regions; (2) it observes the activities of hands almost at all time.  
  
For this homework, we are asked to solve an image classification problem. What we should do is to implement a deep-learning-based approach to determine what things the user takes in hand. 


## Implementation
1. For the file-loading approach, I consider that the normal way to load in all the images will not perform well. For the first reason that the data is in a large size and this may cause a memory insufficient problem. Second, I think that there must be a build-in file-loading method in tensorflow to efficiently load the file. Thus, I found `tfrecord`, which is one of the efficient file-loading approach in tensorflow. It will save the data in a binary file. Then, load in the files with the queue. This can make the data more convenient on storage, copying and movement. Furthermore, since the data will be loaded from the queue, this help as save more memory compared with the fasion that load all images first. Comapared with loading all first, it performs much sufficient and faster, though it is much complicated on implementation and pre-process for the `tfrecord` specific file.

2. Two

```
Code highlights
```

## Installation
* Other required packages.
* How to compile from source?

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


