# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
# import variables

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

# for pr_point
from eval_util import *
# for save data
import numpy as np

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_float('width_multiplier', 1.0,
                            'Width Multiplier, for MobileNet only.')

tf.app.flags.DEFINE_float('gpu_memp', 0.5,
                            'The percentage of gpu memory for use.')
                            

tf.app.flags.DEFINE_string(
    'save_pred', '2', 'The num of pred-cls:{0: fa, 1: ges, 2: obj}')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False,
        width_multiplier=FLAGS.width_multiplier)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label, label1, label2] = provider.get(['image', 'label', 'label1', 'label2'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels, labels1, labels2 = tf.train.batch(
        [image, label, label1, label2],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, logits1, logits2, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)
    predictions1 = tf.argmax(logits1, 1)
    labels1 = tf.squeeze(labels1)
    predictions2 = tf.argmax(logits2, 1)
    labels2 = tf.squeeze(labels2)
    sm_logits2 = tf.nn.softmax(logits2)
    oh_labels2 = slim.one_hot_encoding(labels2, dataset.num_classes[2] - FLAGS.labels_offset)
    oh_labels2 = tf.cast(oh_labels2, dtype=tf.bool)
    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
        'Accuracy1': slim.metrics.streaming_accuracy(predictions1, labels1),
        'Recall_51': slim.metrics.streaming_recall_at_k(
            logits1, labels1, 5),
        'Accuracy2': slim.metrics.streaming_accuracy(predictions2, labels2),
        'Recall_52': slim.metrics.streaming_recall_at_k(
            logits2, labels2, 5),
        'Confusion_matrix': get_streaming_metrics(labels, predictions,
                                                   dataset.num_classes[0] - FLAGS.labels_offset),
        'Confusion_matrix1': get_streaming_metrics(labels1, predictions1,
                                                   dataset.num_classes[1] - FLAGS.labels_offset),
        'Confusion_matrix2': get_streaming_metrics(labels2, predictions2,
                                                   dataset.num_classes[2] - FLAGS.labels_offset),
        'pr_p': get_streaming_curve_points(labels=oh_labels2, predictions=sm_logits2,
                                           num_thresholds=200,curve='PR'),
        'pr_auc': slim.metrics.streaming_auc(predictions=sm_logits2, labels=oh_labels2,
                                             num_thresholds=200,curve='PR'),
    })

    '''
        # Print the summaries to screen.
        for name, value in names_to_values.iteritems():
          summary_name = 'eval/%s' % name
          op = tf.summary.scalar(summary_name, value, collections=[])
          op = tf.Print(op, [value], summary_name)
          tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    '''      
    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    
    ###########################
    #  GPU Memory Use         #
    ###########################
    sconf = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memp) )
    
    # confusion_matrix: row: gt, col: pred
    [confusion_matrix, pr_p, pr_auc] = slim.evaluation.evaluate_once(
    # [logits2, pr_auc] = slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore,
        session_config=sconf,
        final_op=[names_to_updates['Confusion_matrix'+FLAGS.save_pred], 
                  names_to_updates['pr_p'], 
                  names_to_updates['pr_auc']
                  ])
    
    
    # save data
    save_dict = {
        "confusion_matrix": confusion_matrix,
        "pr_point": pr_p,
        "pr_auc": pr_auc,
        "pred_cls": int(FLAGS.save_pred),
    }
    fn = '%s/%s_rst%s.npy'%(FLAGS.eval_dir, FLAGS.dataset_split_name, FLAGS.save_pred)
    np.save(fn, save_dict)
    '''
    print("confusion_matrix")
    print(confusion_matrix)
    # print(type(confusion_matrix))
    print("pr_p")
    print(pr_p)
    # print(type(pr_p))
    print("pr_auc")
    print(pr_auc)
    # print(dir(pr_auc))
    # print(type(pr_auc))
    '''
    # pdb.set_trace()

    
if __name__ == '__main__':
  tf.app.run()
