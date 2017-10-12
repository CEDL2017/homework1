import argparse
import os
import zipfile
import numpy as np
from train_model import *
from test_model import *


# settings for default arguments
parser = argparse.ArgumentParser(description='CEDL2017-homework1')
parser.add_argument('--num-epoch', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--val-split', type=float, default=0.2)
parser.add_argument('--num-fc-neurons', type=float, default=512)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--outdir', type=str, default='output')
parser.add_argument('--run-deepq', type=str, default=False)
args = parser.parse_args()
print(args)


# settings for path
output_path = args.outdir # store output (model) files
if not os.path.exists(output_path):
    os.makedirs(output_path)

if args.run_deepq:
	data_env = os.environ.get('GRAPE_DATASET_DIR')
	frames_path = os.path.join(data_env, 'frames.zip') # video frame image path
	labels_path = os.path.join(data_env, 'labels.zip') # ground truth label path
else:
	frames_path = os.path.join('..', '..', 'frames.zip') # video frame image path
	labels_path = os.path.join('..', '..', 'labels.zip') # ground truth label path

zip_ref_frames = zipfile.ZipFile(frames_path, 'r')
zip_ref_labels = zipfile.ZipFile(labels_path, 'r')

# settings for training
### setting_index = [0, 1]
### mode = ['FA', 'obj', 'ges']
### num_classes = ['FA': 2, 'obj': 24, 'ges': 13]
batch_size = args.batch_size
epochs = args.num_epoch
validation_split = args.val_split
num_fc_neurons = args.num_fc_neurons
dropout_rate = args.dropout

for model_name in ['VGG16']: # ['AlexNet', 'ResNet50', 'VGG16']
	run_train_process(model_name, batch_size, epochs, validation_split, num_fc_neurons, dropout_rate, zip_ref_frames, zip_ref_labels, output_path)
	run_test_process(model_name, batch_size, epochs, num_fc_neurons, dropout_rate, zip_ref_frames, zip_ref_labels, output_path)

zip_ref_frames.close()
zip_ref_labels.close()
