# 丘鈞岳 106065522
> #### Homework 1 : Deep Classification 
## Introduction
In this work, I use the pretrained model( ResNet18), which was pretrained on ImageNet Datasets to train on the given dataset. Consequencely, I test on it and have a good results on my experiments. Why I choose resnet? Cause it's the most polpular NN architecture now. So, I think it will positively much better than the Alexnet baseline.

## Enviroment

* Framework : Pytorch( Python)
* OS : Ubuntu + Linux
* Virtual Evironment : Miniconda2
* Network Architecture : ResNet18

## Implementation

1. I modify my [main.py](https://github.com/pytorch/examples/blob/master/imagenet/main.py) from the pytorch tutorial.
2. And it will load the pretrained ResNet18 automatically.
3. Write a DataLoader( load_dataset.py) to input the corresponding frames and labels for training and testing.
![](https://i.imgur.com/70KI1Qd.png)

4. Setting epoch 3, --batch_size 128 ,--workers 4 on trainig.
![](https://i.imgur.com/4HglBjf.png)




## Result

Testing Accuracy(%)

| 1st epoch | 2nd epoch | 3rd epoch |
| -------- | -------- | -------- |
| 64.152%     | 64.480%     | **66.390%**     |







