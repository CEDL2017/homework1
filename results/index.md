# 王尊玄 <span style="color:red">(106061521)</span>

#Project 1: Deep Classification

## Overview
The project is related to 
> object classification over images captured from dashcam on both hands. There are 24 object classes to be recognized,
```Obj = { 'free','computer','cellphone','coin','ruler','thermos-bottle','whiteboard-pen','whiteboard-eraser','pen','cup','remote-control-TV','remote-control-AC','switch','windows','fridge','cupboard','water-tap','toy','kettle','bottle','cookie','book','magnet','lamp-switch'}```. Our goal is to achieve high accuracy in classifying objects in handcam images.


## Implementation
<center>
<img src="https://1.bp.blogspot.com/-O7AznVGY9js/V8cV_wKKsMI/AAAAAAAABKQ/maO7n2w3dT4Pkcmk7wgGqiSX5FUW2sfZgCLcB/s1600/image00.png" width="600"/>
</center>

1. Reform dataset: as directory structure of the original dataset is quite "abnormal" in classification, we reform the dataset to reformed dataset with diferrent directory structure. This can be done by doing:
```
  # modify train_dir, test_dir, labels_root_dir in ${HW1_ROOT_DIR}/reform_dataset.py
$ python reform_dataset.py
```
2. Convert data to TFrecord: to accelerate training process, we serialize and pack all images to .tfrecord file. This can be done by doing:
```
$ SLIM_DIR=${HW1_ROOT_DIR}/workspace/models/research/slim
  # modify _NUM_VALIDATION in ${SLIM_DIR}/datasets/download_and_convert_CEDLhw1.py
  # modify SPLITS_TO_SIZE in ${SLIM_DIR}/datasets/cedlhw1.py
$ python ${SLIM_DIR}/download_and_convert_data.py --dataset_name=CEDLhw1 \
						  --dataset_dir=${HW1_ROOT_DIR}/data/reformed_train
```
3. Training: We use "Inception-ResNet-v2" network structure whose trained model over ImageNet reaches highest accuracy in tf-slim, as shown in [Pretrained Models section](https://github.com/tensorflow/models/tree/master/research/slim), and finetune on the last fully-connected layer over our own handcam dataset. Pretrained model can be installed from [here](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz). Our training code is modified from [tf-slim training code](https://github.com/tensorflow/models/blob/master/research/slim/train_image_classifier.py), where numerous parameters can be set. There is an easy script that carries out training,
```
  # modify TRAIN_DIR, DATASET_DIR, CHECKPOINT_DIR, SLIM_DIR in ${HW1_ROOT_DIR}/run_training.sh
  # also train_image_classifier_CEDLhw1.py has numerous argument to be set
  # note that SLIM_DIR=${HW1_ROOT_DIR}/workspace
$ source ${HW1_ROOT_DIR}/run_training.sh
```
4. Evaluation: Our evaluation code is modified from [tf-slim evaluation code](https://github.com/tensorflow/models/blob/master/research/slim/eval_image_classifier.py). The following metrics are computed:
	* [accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
	* [precision-recall plot](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
	* [confusion matrix](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
```
  # modify TRAIN_DIR, CHECKPOINT_DIR, DATASET_DIR, SLIM_DIR in ${HW1_ROOT_DIR}/run_training.sh
  # note that SLIM_DIR=${HW1_ROOT_DIR}/workspace and RESULTS_DIR is not used
$ source ${HW1_ROOT_DIR}/run_eval.sh
```

## Installation
* Python
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim)
* [sklearn](http://scikit-learn.org/stable/)

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


