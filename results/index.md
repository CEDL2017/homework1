# Guan-Yuan Chen <span style="color:red">(105065530)</span>

## Overview
In this project, I used Inception-ResNet-v2 and ResNet-152 as fixed feature extractors (no finetune) to experiment and analysis the different methods for classification. <br> 
Include:<p>

- [`ResNet-152_FE_batch256.py`](https://github.com/guan-yuan/homework1/blob/master/ResNet-152_FE_batch256.py): A model that uses the ResNet-152 as the fixed feature extractor with data augmentation (resize the short side to 256 then crop to 224) on hand and head images separately.<p>
	
- [`ResNet-152_FE_batch256_crop232.py`](https://github.com/guan-yuan/homework1/blob/master/ResNet-152_FE_batch256_crop232.py): A model that uses the ResNet-152 as the fixed feature extractor with data augmentation (resize the short side to 232 then crop to 224) on hand and head images separately.<p>
	
- [`ResNet-152_FE_batch256_crop224.py`](https://github.com/guan-yuan/homework1/blob/master/ResNet-152_FE_batch256_crop224.py): A model that uses the ResNet-152 as the fixed feature extractor with no data augmentation on hand and head images separately.<p>

- [`Two_stream_InceptionResNetV2_FE_batch64_dropout03.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_InceptionResNetV2_FE_batch64_dropout03.py): A model that uses the InceptionResNetV2 as the fixed feature extractor with data augmentation (resize the short side to 312 then crop to (299, 299)) for the two stream way.<p>
	
- [`Two_stream_ResNet-152_FE_batch64_dropout03.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_ResNet-152_FE_batch64_dropout03.py): A model that uses the ResNet-152 as the fixed feature extractor with data augmentation (resize the short side to 256 then crop to 224) for the two stream way.<p>
	
- [`Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout03.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout03.py): A model that uses the ResNet-152_with_InceptionResNetV2 as the fixed feature extractor with data augmentation (resize the short side to 312 then crop to (299, 299)) for the two stream way.<p>
	
- [`Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01.py`](https://github.com/guan-yuan/homework1/blob/master/Two_stream_ResNet-152_with_InceptionResNetV2_FE_batch32_dropout01.py): A model that uses the ResNet-152_with_InceptionResNetV2 as the fixed feature extractor with data augmentation (resize the short side to 312 then crop to (299, 299)) for the two stream way.<p>
	
Due to the considerations of computational resources and efficiency (DeepQ do not support the pytorch), I just use fixed feature extractor method for the classification task. And I pay more attention to the data augmentation issues and the  discrepancy between the different features that extracted from different CNN models and how to use that features to reach higher performance (accuracy).

## Reference
[Recognition from Hand Cameras: A Revisit with Deep Learning](https://drive.google.com/file/d/0BwCy2boZhfdBM0ZDTV9lZW1rZzg/view) <p>
[Pytorch Transfer Learning tutorial](http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) <p> 
[Pretrained models for Pytorch](https://github.com/Cadene/pretrained-models.pytorch/blob/master/README.md) <p>
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) <p>
[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) <p>


## Implementation
1. One
	* item
	* item
2. Two

```
Code highlights
```

## Installation
* Other required packages.
* How to compile from source?

### Results
<div align=left>
<img src="https://github.com/guan-yuan/homework1/blob/master/output/two_stream_testing.png" width = "80%" alt="hierarchical softmax computations"/>
</div>



