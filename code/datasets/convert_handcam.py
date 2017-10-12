# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import numpy as np

import tensorflow as tf

from datasets import dataset_utils

'''
python -c "from datasets import convert_handcam as ch; ch.run('/home/nvlab/cedl/hw01',cl=3)"
'''
# The number of images in the validation set.
_NUM_VALIDATION = 12776

_FILE_PAT = ['handcamfa_%s_%05d-of-%05d.tfrecord', 
             'handcamges_%s_%05d-of-%05d.tfrecord', 
             'handcamobj_%s_%05d-of-%05d.tfrecord',  
             'handcam_%s_%05d-of-%05d.tfrecord']
# The number of shards per dataset split.
_NUM_SHARDS = 5

Fa = { 'free':0,
       'active':1}

Obj = { 'free':0,
        'computer':1,
        'cellphone':2,
        'coin':3,
        'ruler':4,
        'thermos-bottle':5,
        'whiteboard-pen':6,
        'whiteboard-eraser':7,
        'pen':8,
        'cup':9,
        'remote-control-TV':10,
        'remote-control-AC':11,
        'switch':12,
        'windows':13,
        'fridge':14,
        'cupboard':15,
        'water-tap':16,
        'toy':17,
        'kettle':18,
        'bottle':19,
        'cookie':20,
        'book':21,
        'magnet':22,
        'lamp-switch':23}

Ges= {    'free':0,
          'press':1,
          'large-diameter':2,
          'lateral-tripod':3,
          'parallel-extension':4,
          'thumb-2-finger':5,
          'thumb-4-finger':6,
          'thumb-index-finger':7,
          'precision-disk':8,
          'lateral-pinch':9,
          'tripod':10,
          'medium-wrap':11,
          'light-tool':12}


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_png(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_png(self, sess, image_data):
        image = sess.run(self._decode_png,
                                         feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir, cl=0, train=False):
    """Returns a list of filenames and inferred class names.

    Args:
        dataset_dir: A directory containing a set of subdirectories representing
            class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
        A list of image file paths, relative to `dataset_dir` and the list of
        subdirectories, representing class names.
    """
    handcam_root = dataset_dir
    cls = ['FA', 'ges', 'obj']
    place_num = [3,3,4,4,3,3]
    lbl_path_pats = ['labels/house/%s_left%d.npy',
                     'labels/house/%s_right%d.npy',
                     'labels/lab/%s_left%d.npy',
                     'labels/lab/%s_right%d.npy',
                     'labels/office/%s_left%d.npy',
                     'labels/office/%s_right%d.npy']
    data_path_pats = ['frames/%s/house/%d/Lhand/Image%d.png',
                      'frames/%s/house/%d/Rhand/Image%d.png',
                      'frames/%s/lab/%d/Lhand/Image%d.png',
                      'frames/%s/lab/%d/Rhand/Image%d.png',
                      'frames/%s/office/%d/Lhand/Image%d.png',
                      'frames/%s/office/%d/Rhand/Image%d.png',]
    if train:
        tr_str = 'train'
        num_bias = 0
    else:
        tr_str = 'test'
        num_bias = 1
    if cl < 3:
        indlbl = None
        photo_filenames = []
        for i, lbl_path_pat in enumerate(lbl_path_pats):
            for j in xrange(1, place_num[i]+1):
                # load lbl
                indtmp = np.load(os.path.join(handcam_root, lbl_path_pat% (cls[cl], j+num_bias*place_num[i]) ))
                # print("Load: (%d, %d), len: %d"%(i, j, indtmp.shape[0]))
                if i==0 and j==1:
                    indlbl = indtmp
                else:
                    indlbl = np.hstack((indlbl, indtmp))
                # load data path
                for k in range(1, indtmp.shape[0]):
                    photo_filenames.append(os.path.join(handcam_root, data_path_pats[i] % (tr_str, j, k) ))
        indlbl = np.array(indlbl, dtype=np.int16)
    else:
        indlbl = []
        photo_filenames = []
        for i, lbl_path_pat in enumerate(lbl_path_pats):
            for j in xrange(1, place_num[i]+1):
                for c in xrange(cl):
                    # load lbl
                    indtmp = np.load(os.path.join(handcam_root, lbl_path_pat% (cls[c], j+num_bias*place_num[i]) ))
                    # print("Load: (%d, %d), len: %d"%(i, j, indtmp.shape[0]))
                    if i==0 and j==1:
                        indlbl.append(indtmp)
                    else:
                        indlbl[c] = np.hstack((indlbl[c], indtmp))
                # load data path
                for k in range(1, indtmp.shape[0]):
                    photo_filenames.append(os.path.join(handcam_root, data_path_pats[i] % (tr_str, j, k) ))
        for c in xrange(cl):
            indlbl[c] = np.array(indlbl[c], dtype=np.int16)

    return photo_filenames, indlbl


def _get_dataset_filename(dataset_dir, split_name, shard_id, cl=0):
    output_filename = _FILE_PAT[cl] % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, indlbl, dataset_dir, cl=0):
    """Converts the given filenames to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
            (integers).
        dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    if cl < 3:
        num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))        
        with tf.Graph().as_default():
            image_reader = ImageReader()
            
            with tf.Session('') as sess:
            
                for shard_id in range(_NUM_SHARDS):
                    output_filename = _get_dataset_filename(
                            dataset_dir, split_name, shard_id, cl=cl)

                    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                        start_ndx = shard_id * num_per_shard
                        end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                        for i in range(start_ndx, end_ndx):
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                    i+1, len(filenames), shard_id))
                            sys.stdout.flush()

                            # Read the filename:
                            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                            height, width = image_reader.read_image_dims(sess, image_data)

                            class_id = indlbl[i]

                            example = dataset_utils.image_to_tfexample(
                                    image_data, 'png', height, width, class_id)
                            tfrecord_writer.write(example.SerializeToString())

    else:
        num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
        with tf.Graph().as_default():
            image_reader = ImageReader()
            
            with tf.Session('') as sess:
            
                for shard_id in range(_NUM_SHARDS):
                    output_filename = _get_dataset_filename(
                            dataset_dir, split_name, shard_id, cl=cl)

                    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                        start_ndx = shard_id * num_per_shard
                        end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                        for i in range(start_ndx, end_ndx):
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                    i+1, len(filenames), shard_id))
                            sys.stdout.flush()

                            # Read the filename:
                            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                            height, width = image_reader.read_image_dims(sess, image_data)

                            class_id = [indlbl[0][i], indlbl[1][i], indlbl[2][i]]

                            example = dataset_utils.image_to_tfexample(
                                    image_data, 'png', height, width, class_id)
                            tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()



def _dataset_exists(dataset_dir, cl=0):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id, cl=cl)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_dir, cl=0):
    """Runs the download and conversion operation.

    Args:
        dataset_dir: The dataset directory where the dataset is stored.
    """
    cl2id = None
    if cl==0:
        cl2id = Fa
    elif cl==1:
        cl2id = Ges
    elif cl==2:
        cl2id = Obj
    elif cl==3:
        cl2id = ['Fa', 'Ges', 'Obj']
    cls = ['FA', 'ges', 'obj']
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir, cl=cl):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    training_filenames, trlbl = _get_filenames_and_classes(dataset_dir, cl=cl, train=True)
    validation_filenames, vallbl = _get_filenames_and_classes(dataset_dir, cl=cl, train=False)


    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, trlbl, dataset_dir, cl=cl)
    _convert_dataset('validation', validation_filenames, vallbl, dataset_dir, cl=cl)

    # Finally, write the labels file:
    if cl < 3:
        labels_to_class_names = {y:x for x,y in cl2id.iteritems()}
        dataset_utils.write_label_file(labels_to_class_names, dataset_dir, filename=cls[cl]+'lbl.txt')
    else:
        for i in xrange(cl):
            filename=cls[i]+'lbl.txt'
            if tf.gfile.Exists(os.path.join(dataset_dir, filename)):
                continue
            labels_to_class_names = {y:x for x,y in cl2id[i].iteritems()}
            dataset_utils.write_label_file(labels_to_class_names, dataset_dir, filename=filename)

    print('\nFinished converting the handcam dataset!')

