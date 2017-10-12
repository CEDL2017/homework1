# 翁正欣 <span style="color:red">106062577</span>

#Project 5: Deep Classification

## Overview
In this project, I implement a model which want to classify objects in a frame captured from video frames, the input is single image, and the output is hand state, object and gesture in image. 
* `test_model.ipynb`: test trained CNN model
* `train_CNN.py`: train CNN model

## Implementation
### Network architecture
![](https://i.imgur.com/18XnIuY.png)
### Training 
in all training time, the VGG19 are fixed. The image is randomly fliped in each epoch to make the model generalize well.

## Installation
* required package: tensorflow
* download [VGG19 weights](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) from [this repository](https://github.com/machrisaa/tensorflow-vgg) and put under parent directory.
* `python train_CNN.py` to train CNN model.

directory tree will look like this:
<pre>
parent_dir/
|-- frames/  
|-- labels/
|-- vgg19.npy
`-- this_repository
</pre>
### Results

<table border=1>
  <tr>
    <th></th>
    <th>free v.s. active</th>
    <th>gesture</th> 
    <th>object</th>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.641672</td>
    <td>0.387289</td>
    <td>0.352301</td>
  </tr>
<!-- 
<tr>
<td>

</td>
</tr> -->

</table>


