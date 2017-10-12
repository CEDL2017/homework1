# 鄭安傑 <span style="color:red">(103062108)</span>

#Project 1: Deep Classification

## Overview
The project is to implement a deep-learning-based method to recognize `object categories` and `free/active` in hand states using transfer learning.

## Implementation
1. **Single-stream ConvNet.** Utilizing only HandCam. 

2. **Number of Trainable Layers.** In transfer learning, it is possible to fine-tune all the layers of the ConvNet, or to keep only some of the earlier layers fixed and only fine-tune some higher-level layers of the network. Several pilot experiments are performed to observe the trend, and found fine-tuning all the layers is hard to converge and has overfitting concerns. As a result, the accuracy of tunning all layers is approximately `5%` lower than only fine-tuning the three fully-connected layers. Thus, to avoid overfitting and save time, we first train only the last three FC layers for 50 epochs, then we train the whole network for another 10 epochs.
3. **Two-streams ConvNet.** Utilizing both HandCam and HeadCam. Dropout layers are insert between FC layers to avoid overfitting.

	* AlexNet
![](https://i.imgur.com/xTe4D4N.png)
	* VGG16
![](https://i.imgur.com/eJ1dBTP.png)


4. **Data Augmentation.**
	* Horizontal Flip     `tf.image.random_flip_left_right(images)`
    * Brightness `tf.image.random_brightness(images, max_delta=0.3)`
    * Contrast `tf.image.random_contrast(images, 0.8, 1.2)`

## Dev Environment
Developed on PC with:
- Ubuntu 16.04
- Python 3.6
- intel i5-4460
- Nvidia GeForce GTX 1080-Ti GPU
- Cuda 8.0


## Installation
* Download [image list](https://drive.google.com/file/d/0B-MtVXQMUxQiVzZiZmZOZy0talE/view?usp=sharing) for data input.
* Download [AlexNet](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/), and [VGG16 ](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) pretrained weights, and correctly set the filepath.
* `python train.py` to train the model.A GPU is recommended for this step. Press `Ctrl+C` then the process will write the weights to checkpoints and return.
* Usage is as follows:
1. `--batch_size`: Set batch size, otherwise defalut `128`.
2. `--learning_rate`: Set learning rate, otherwise defalut `0.0001`.
3. `--num_epochs`: Set number of epochs, otherwise defalut `200`.
4. `--dropout_rate`: Set dropout rate, otherwise defalut `0.5`.
5. `--usingVGG`: Set `True` to use `VGG16`, otherwise defalut using `AlexNet`.

## Results
1. **Free/Active.** The left figure below shows that VGG16 generally outperforms AlexNet on the dataset. Moreover, 2-Stream networks taking advantage of both HandCam and HeadCam have better accuracies. Due to time constraints, only 5 epochs are trained. As the right figure shows, after traing 2-Stream VGG16 for 50 epochs, the accuracy reaches `0.7328` and still indicates an increasing trend.
![](https://i.imgur.com/nalHThp.png)![](https://i.imgur.com/gLXUxR7.png)
     
2. **Object Categories.** As the left figure shows , dropout layers provide a simple way to avoid overfitting. We kept most of the earlier layers fixed and train 50 epochs, reaching the accuracy around `0.6350`. Then we retrain all layers for another 10 epochs and the accuracy reaches `0.6434`. Noted that while training all layers, it's hard to converge and cost a lot of time. 
![](https://i.imgur.com/0yCBctx.png)![](https://i.imgur.com/Yr2T15s.png)

    A screenshot of the Tensorboard visualizing training accuracy and loss.
![](https://i.imgur.com/9gWgRaj.png)

    This is the heatmap of the final classification confusion matrix, where each alphabet corresponds to one category. As the figure shows, the best predicted categry is `A`, `H`, and `G`, which is `free`, `whiteboard-eraser`, and `whiteboard-pen`, respectively. The worst predicted categry is `Q`, `M`, and `Y`, which is `water-tap`, `switch`, and `lamp-switch`, respectively.
![](https://i.imgur.com/YsXvVaM.png)

    While in precision-recall, the best classes are `free`, `whiteboard-pen`, and `cupboard`. The worst  classes are `magnet` `switch` `ruler`.
![](https://i.imgur.com/2N3x729.png)

    The Average precision score(AUC), micro-averaged over all classes: AUC=`0.64`
![](https://i.imgur.com/gwxL6c2.png)



## Download Model
The final weight can be downloaded from [here](https://drive.google.com/file/d/0B-MtVXQMUxQibjNqZDlxNlpENG8/view?usp=sharing) ( ~1GB).

