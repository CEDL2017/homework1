# CEDL2017 HW1 Report
Author: Howard Lo (羅右鈞) 105062509

## Project Overview
In this project, we implemented mainly three kinds of model for classifying objects in video frames with TensorFlow:
- [`hand_obj_vgg_16.ipynb`](https://github.com/YuChunLOL/homework1/blob/master/hand_obj_vgg_16.ipynb): A finetuned VGG-16 model trained on object classification task.
- [`hand_gesture_vgg_16.ipynb`](https://github.com/YuChunLOL/homework1/blob/master/hand_gesture_vgg_16.ipynb): A finetuned VGG-16 model trained on hand gesture classification task.
- [`two_stream_vgg_16_baseline.ipynb`](https://github.com/YuChunLOL/homework1/blob/master/two_stream_vgg_16_baseline.ipynb): A two-stream VGG-16 model based on our finetuned `hand_obj_vgg_16` and `hand_gesture_vgg_16` models.
- [`two_stream_vgg_16_multi_loss.ipynb`](https://github.com/YuChunLOL/homework1/blob/master/two_stream_vgg_16_multi_loss.ipynb): Similar to  our `two_stream_vgg_16_baseline` model, but instead of only being trained on object classfication task, we jointly trained the model on both object and hand gesture classfication tasks.

Each notebook above contains the details of model implementation, training and evaluation (only cross-entropy loss and accuracy). See more evaluations such as precision-recall curve and confusion matrix in [`visualize_performance.ipynb`](https://github.com/YuChunLOL/homework1/blob/master/visualize_performance.ipynb).

Other files:
- [`data_helper.py`](https://github.com/YuChunLOL/homework1/blob/master/data_helper.py): A set of helper functions for loading dataset. You should first download [dataset](https://drive.google.com/drive/folders/0BwCy2boZhfdBdXdFWnEtNWJYRzQ)(`frames/` and `labels/`) and place them to `dataset/`.
- [`resize_image.ipynb`](https://github.com/YuChunLOL/homework1/blob/master/resize_image.ipynb): Resize images as preprocessing step. It will save processed images in `dataset/resize/` folder.
- [`vgg_preprocessing.py`](https://github.com/YuChunLOL/homework1/blob/master/vgg_preprocessing.py): Preprocess(random cropping, flipping, etc.) images to standard input of VGG-16 net, used in TensorFlow input piplines.
- [`inspect_checkpoint.py`](https://github.com/YuChunLOL/homework1/blob/master/inspect_checkpoint.py): For debugging, we can inspect TensorFlow checkpoints by running:
`$ python inspect_checkpoint.py --file_name=<checkpoint_name>`

## Dev Environment
This project is developed on our lab server with:
- OS: Ubuntu 16.04.3 LTS
- Anaconda 4.3.28 (`conda` for package management and virtual environment management)
- Python 3.6.2
- 20 core Intel(R) Xeon(R) CPU E5-2630
- 2 Nvidia GeForce GTX 1080-Ti GPUs
- Nvidia cuDNN 6.0.21
- Cuda 8.0

## Setup
1. Create a virtaul environment using `conda` by running:
`$ conda create --name <env_name> --file conda_requirements.txt`
2. Activate your virtual environment:
`$ source activate <env_name>`
3. Install additional python packages by running:
`$ pip install -r pip_requirements.txt`
4. Run jupyter notebook to view our `.ipynb` notebook files:
`$ jupyter notebook`
5. Remember to download [dataset](https://drive.google.com/drive/folders/0BwCy2boZhfdBdXdFWnEtNWJYRzQ)(`frames/` and `labels/`) and place them to our dataset folder `dataset/`.
6. You may want to resize the video frames(1920x1080) in advance to make training faster and lower memory usage by running [`resize_image.ipynb`](https://github.com/YuChunLOL/homework1/blob/master/resize_image.ipynb).
7. Download pretrained VGG-16 net and place it to our model folder `model/`:
```
$ cd model/
$ wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
$ tar -xvf vgg_16_2016_08_28.tar.gz
$ rm vgg_16_2016_08_28.tar.gz
```

Note: See detailed requirements in [`conda_requirements.txt`](https://github.com/YuChunLOL/homework1/blob/master/conda_requirements.txt) and [`pip_requirements.txt`](https://github.com/YuChunLOL/homework1/blob/master/pip_requirements.txt).

## Implementation

### Model Architectures
For image classification, the common way is first, grab a  CNN model pretrained on ImageNet, such as AlexNet, VGG-16, ResNet, etc. Secondly, finetune the pretrained model on our own dataset. This is also known as ["Transfer Learning"](http://cs231n.github.io/transfer-learning/), which has several advantages comparing to training a model from scratch (randomly initialized model parameters):
1. The earlier layers of pretrained model have already contained generic low-level features (e.g. edge, corner and color features), which helps our models converge faster when training on a new dataset, instead of learning from scratch.
2. Since our new dataset is very imbalanced and relatively small, if we train our model from scratch, it is likely to become easily overfitting, where the earlier layers may just learn dataset-specific features.

Thus, we use pretrained VGG-16 net regarding to its feasible performance and model size as our building block of our models.
#### Model 1: `hand_obj_vgg_16` and `hand_gesture_vgg_16`
The model architecture of both models is same as the standard VGG-16 net architecture. Please refer to [vgg.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py#L132) in TensorFlow.

#### Model 2: `two_stream_vgg_16_baseline`
Once we finish training both `hand_obj_vgg_16` and `hand_gesture_vgg_16` models, we then load the weights respectively to their corresponding streams(blue one and red one in the plot). The intuition behind the design is that, the object grabbed in hand usually has a high correlation with hand gesture itself. Based on the intuition, we classfiy object not only consider the deep features of object outputted from `hand_obj_vgg_16` model(VGG-16 net excluding the last two layers) , but also the deep features of hand gesture outputted from `hand_gesture_vgg_16`(VGG-16 net excluding the last two layers). This design actually boosts **7.5%** accuracy compared to `hand_obj_vgg_16` model.
![](https://i.imgur.com/lkR0twR.png)

#### Model 3: `two_stream_vgg_16_multi_loss`
We also tried **multi-task learning** using two stream model. Instead of just only trained on object classification task, we trained this model on both object classfication task and hand gesture classification task. That is, during training, we jointly minimize the sum of cross-entroy losses of this 2 taskes. The motivation behind is that multi-task learning can make model learn more generalized representation and prevent overfitting [1]. Although the accuracy is about **1%** lower than the `two_stream_vgg_16_baseline` model, which is what we expected (since the model now needs to learn two tasks simultaneously, the model could lose a little bit performance from the main task), we observed that the overfitting situation on training set during training `two_stream_vgg_16_multi_loss` model was **much less severe** than training `two_stream_vgg_16_baseline` model.

Furthermore, we thought that simply summing up the object classfication loss (main task) and hand gesture classification loss (auxiliary task) may be too naive, and also may be the reason causing accuracy loss on the main task. Therefore, we may probabily try to adopt **weighted loss** [2], which is introduced in this year,  but we did not implement this idea due to the time limitation. This may be served as future work.

![](https://i.imgur.com/D7w1kEN.png)

See references:
- [1] [An Overview of Multi-Task Learning in Deep Neural Networks](http://ruder.io/multi-task/)
- [2] [Multi-Task Learning Using Uncertainty to Weigh
Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115.pdf)

### Training Details
Here, we will describe how we perform the dataset splitting, data augmentation, training procedures, optimization and hyperparameter settings.

#### Dataset splitting
We use `data_helper.load_dataset()` to load our training, validation and testing set. The data size of each set:
- Training set: 12744
- Validation set: 2248
- Testing set: 12776

Note that we did not shuffle video frames before splitting the original training data to training set and validation set due to the characteristic of video. If we shuffle video frames before we perform splitting, then the validation accuracy will be inaccurate, since the video frames will be very similar in training set and validation set.

#### Data augmentation
During training, the image is randomly cropped and flipped in every training epoch. This can help our model to generalize well. Also, it is common to perform downsampling for majority class or upsampling for minority class when the dataset is imbalanced, but we're interested in model architectures in this project, so we mainly focused on designing model architectures instead of spending too much time to augment our dataset.

#### Training procedure
We have two stages in training:
1. At first stage:
    - For `hand_obj_vgg_16` and `hand_gesture_vgg_16`, we only train the last layer("fc8").
    - For `two_stream_vgg_16_baseline` and `two_stream_vgg_16_multi_loss`, we only train the last layer("fc8") and the second last layer("fc7").
2. At second stage, we train whole layers for all models.

#### Optimization
During training, we performed a set of optimization tricks, such as:
- Early stopping to prevent the model from training for too long time and becoming overfitting on the training set.
- Apply dropout layers in our models.
- Using L2 Regularization loss.

#### Hyperparameters
- Batch size:
    - **32** for `hand_obj_vgg_16` and `hand_gesture_vgg_16`.
    - **16** for `two_stream_vgg_16_baseline` and `two_stream_vgg_16_multi_loss` due to the GPU memory limitation.
- Optimizer:
    - **Gradient descent optimizer** for `hand_obj_vgg_16` and `hand_gesture_vgg_16`.
    - **Adam optimizer** for `two_stream_vgg_16_baseline` and `two_stream_vgg_16_multi_loss`.
- Learning rate:
    - **1e-3** for training stage 1.
    - **1e-5** for training stage 2.
- Weight decay: **5e-4**
- Dropout rate: **0.5**
- Regularizer: **L2**
- Patience epoch number for early stopping: **5**

## Results

### Testing accuracy on object classfication task.
| Model | Accuracy |  
|-------|----------|
| `hand_obj_vgg_16`| 58.1% |
| `two_stream_vgg_16_baseline`| **65.6%** |
| `two_stream_vgg_16_multi_loss`| 64.0% |
Note: The auxiliary model `hand_gesture_vgg_16` has accuracy of 61.9% on hand gesture classfication task.

### Precision-Recall Curve and Confusion Matrix
Please refer to [`visualize_performance.ipynb`](https://github.com/YuChunLOL/homework1/blob/master/visualize_performance.ipynb).

#### Model 1: `hand_obj_vgg_16`
Precision-Recall Curve:
![](https://i.imgur.com/g392Yz5.png)
Confusion Matrix:
![](https://i.imgur.com/ZQGBeuh.png)

#### Model 2: `two_stream_vgg_16_baseline`
Precision-Recall Curve:
![](https://i.imgur.com/7sxVahz.png)
Confusion Matrix:
![](https://i.imgur.com/C0DQ2CM.png)

#### Model 3: `two_stream_vgg_16_multi_loss`
Precision-Recall Curve:
![](https://i.imgur.com/M3R0Mty.png)
Confusion Matrix:
![](https://i.imgur.com/etClzSq.png)

#### Discussion: Dataset problem
Since the dataset is very imbalanced, we found that the reason why the AUC value
is not similar to accuracy is because we splitted validation set from training data to do early stopping
This operation caused the training set lacking of some of minority classes, which leaded to
the low value of AUC. We can address this by directly training on entire training data without splitting extra validation set, or we can split validation set class-by-class carefully. (Our's splitting method was just splitting 2 portion of training video frames into training set and validation set)
