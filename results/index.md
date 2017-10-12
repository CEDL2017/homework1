# Guan-Yuan Chen <span style="color:red">(105065530)</span>

## Overview
In this project, I used Inception-ResNet-v2 and ResNet-152 as fixed feature extractors (no fine-tune) to experiment and analysis the different methods for classification. <br> 
Include:<p>

- [`ResNet-152_FE_batch256.py`](https://github.com/guan-yuan/homework1/blob/master/ResNet-152_FE_batch256.py): A model that uses the ResNet-152 as the fixed feature extractor with data augmentation (resize the short side to 256 then crop to 224 X 398) on hand and head images separately.<p>
	
- [`ResNet-152_FE_batch256_crop232.py`](https://github.com/guan-yuan/homework1/blob/master/ResNet-152_FE_batch256_crop232.py): A model that uses the ResNet-152 as the fixed feature extractor with data augmentation (resize the short side to 232 then crop to 224 X 398) on hand and head images separately.<p>
	
- [`ResNet-152_FE_batch256_crop224.py`](https://github.com/guan-yuan/homework1/blob/master/ResNet-152_FE_batch256_crop224.py): A model that uses the ResNet-152 as the fixed feature extractor with no data augmentation on hand and head images separately.<p>

- [`Two_stream_InceptionResNetV2_FE_batch64_dropout03.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_InceptionResNetV2_FE_batch64_dropout03.py): A model that uses the InceptionResNetV2 as the fixed feature extractor with data augmentation (resize the short side to 312 then crop to (299, 299)) for the two streams way.<p>
	
- [`Two_stream_ResNet-152_FE_batch64_dropout03.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_ResNet-152_FE_batch64_dropout03.py): A model that uses the ResNet-152 as the fixed feature extractor with data augmentation (resize the short side to 256 then crop to 224 X 398) for the two streams way.<p>
	
- [`Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout03.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout03.py): A model that uses the ResNet-152_with_InceptionResNetV2 as the fixed feature extractor with data augmentation (resize the short side to 312 then crop to (299, 299)) for the two streams way.<p>
	
- [`Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01.py): A model that uses the ResNet-152_with_InceptionResNetV2 as the fixed feature extractor with data augmentation (resize the short side to 312 then crop to (299, 299)) for the two streams way.<p>
	
Due to the considerations of computational resources and efficiency (DeepQ do not support the pytorch), I just use the fixed feature extractor method for the classification task. And I pay more attention to the data augmentation issues and the  discrepancy between the different features that extracted from different CNN models and how to use those features to reach higher performance (accuracy).

## Reference
[Recognition from Hand Cameras: A Revisit with Deep Learning](https://drive.google.com/file/d/0BwCy2boZhfdBM0ZDTV9lZW1rZzg/view) <p>
[Pytorch Transfer Learning tutorial](http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) <p> 
[Pretrained models for Pytorch](https://github.com/Cadene/pretrained-models.pytorch/blob/master/README.md) <p>
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) <p>
[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) <p>


## Implementation
1. One: The first type of implementation is using hand and head images separately for classification. In this step, I focus on analysis the different effect on accuracy of using different models and different data augmentation skills to extract features and the accuracy results of classification problems.<p>
[`ResNet-152_FE_batch256.py`](https://github.com/guan-yuan/homework1/blob/master/ResNet-152_FE_batch256.py)<p>
[`ResNet-152_FE_batch256_crop232.py`](https://github.com/guan-yuan/homework1/blob/master/ResNet-152_FE_batch256_crop232.py)<p>
[`ResNet-152_FE_batch256_crop224.py`](https://github.com/guan-yuan/homework1/blob/master/ResNet-152_FE_batch256_crop224.py)<p>



2. Two: The second type of implementation is using hand and head images jointly for classification. As the paper does, the method (two streams) can help to cover the shortages of only using hand or head images separately. In this step, I focus on analysis the different effect on accuracy of using different models for the two streams classification method.<p>
[`Two_stream_InceptionResNetV2_FE_batch64_dropout03.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_InceptionResNetV2_FE_batch64_dropout03.py)<p>
[`Two_stream_ResNet-152_FE_batch64_dropout03.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_ResNet-152_FE_batch64_dropout03.py)<p>


3. Three: By the further consideration that different architecture of CNN models have different abilities and performances on extract the head and hand images. In order to combine the capacity of different models, I use the **double two streams** method to extract the head and hand images via the two different models and then connect those features for classification. In this experiment, the method has a remarkable promotion on the final top 1 accuracy.<p>
[`Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout03.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout03.py)<p>
[`Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01.py)<p>
<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/models.png" width = "100%" alt=""/>
</div>


## Requirements
pytorch v0.2.0<p>
torchvision<p>
anaconda default packages<p>

## How to use
1. Unzip the dataset to the the "./data" folder.
2. ```python "the_model_you_want_to_train(test)"```<br>
e.g. ```python Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01.py```

## Access trained parameters and records
[Link](https://drive.google.com/drive/folders/0B4-rB9HD2WbEQUQ3SHpoWlhOMEE)
Download from the "Link", then replace the "./save" folder.

### Results
1. In the project, I experiment different random cropped size on images for evaluating the effect of data augmentation. The following images are the example that hand and head images randomly cropped to 299 x 299 (resize the short side to 312 then crop to (299, 299)).<p>

Hand images
<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01_train_batch_hand.png" width = "100%" alt=""/>
</div>
<p>
<p>
<p>
Head images
<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01_train_batch_head.png" width = "100%" alt=""/>
</div>

The results showed below explain that the different cropped sizes have small effect on both hand and head images, but the models with no crop images have the highest accuracy on classification task and when the cropped range increases the accuracy may decline.  

<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/diff_DA_training.png" width = "50%" alt=""/>
</div>


<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/diff_DA_testing.png" width = "50%" alt=""/>
</div>

2. The results showed below display the performance on Inception-ResNet-v2 and ResNet-152 models using hand and head images  separately (for training and testing).

<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/one_stream_training.png" width = "50%" alt=""/>
</div>

<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/one_stream_testing.png" width = "50%" alt=""/>
</div>

3. The results showed below display the performance on two streams and **double two streams models** training and testing on hand and head images jointly. Interestingly, although the ResNet-152 has the obvious higher accuracy on hand images task (test), at the two streams case the Inception-ResNet-v2 outperform than ResNet-152 (test). And we can find that, the ResNet-152 model prone to over-fitting than Inception-ResNet-v2.

<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/two_stream_training.png" width = "50%" alt=""/>
</div>

<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/two_stream_testing.png" width = "50%" alt=""/>
</div>

### The Best Top1 Testing Accracy
| Model | Accuracy |  
|-------|----------|
| `Two_stream_ResNet-152_FE_batch64_dropout03`| 0.662% |
| `Two_stream_InceptionResNetV2_FE_batch64_dropout03`| 0.708% |
| `Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout03`| 0.712% |
| `Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01`| **0.717%** |


