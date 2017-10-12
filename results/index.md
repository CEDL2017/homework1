# Project 5: Deep Classification
#### 吳奕萱 <span style="color:red">106062581</span>

## Overview
The project is related to VGG-Net
> https://arxiv.org/pdf/1409.1556.pdf


## Implementation
1. Convolutions
	* Six Convolutional Layers and additional pooling layers after 2 layer of convolution
        - Layer 1 Convolution: 16 5x5 filter
        - Layer 2 Convolution: 32 5x5 filter
        - Layer 3 Pooling: Down pooling to half tensor size
        - Layer 4 Convolution: 64 5x5 filter
        - Layer 5 Convolution: 128 5x5 filter
        - Layer 6 Pooling: Down pooling to half tensor size
        - Layer 7 Convolution: 256 5x5 filter
        - Layer 8 Convolution: 128 5x5 filter
        - Layer 9 Pooling: Down pooling to half tensor size
2. Full Connected Layers
    * Three Layers of FC for each label (FA, ges, obj)
        - Layer 10 Full Connected: layer9 tensor size -> 1000
        - Layer 11 Full Connected: 1000 -> 100
        - Layer 12 Full Connected: 100 -> Number of label class
            - FA: 2
            - Ges: 13
            - Obj: 24
3. Prediciton
    * Softmax layer is added after FC layers

4. Loss Function and Optimization
    * Cross Entropy
    * Adam Optimizator

## Installation
* Other required packages.
    No
* How to compile from source?

    Testing Model (CEDL_homework1.ipynb)
    1. Run First block to import package
        - NOTE: 2 GPUs are assigned in this part. Therefore, GPUs need to be assigned to testing the model.
    2. Run Load Data part in the notebook
        - NOTE: The data path need to be re-assigned before testing.
    3. Run Model part in the notebook
    3. Run Testing blocks

### Results
 * NOTE: My model hasn't converged yet
 * The validation accuracy is 0.64
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