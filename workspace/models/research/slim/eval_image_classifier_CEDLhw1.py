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

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

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

FLAGS = tf.app.flags.FLAGS

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion_matrix",
                          cmap=plt.cm.jet):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt), 
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predictd label")

    plt.savefig("/home/johnson/Desktop/CEDL/homework1/confusion_matrix.jpg")

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
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
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

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

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

    saver = tf.train.Saver(variables_to_restore)

    ##########################################
    ##              run prediction          ##
    ##########################################
    predictions_list = []
    labels_list = []
    logits_list = []
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver.restore(sess, checkpoint_path)
        for batch_id in range(int(num_batches)):
            predictions_, labels_, logits_ = sess.run([predictions,labels,logits])
            predictions_list.extend(predictions_)
            labels_list.extend(labels_)
            logits_list.extend(logits_)

        coord.request_stop()
        coord.join(threads)
        sess.close()

        predictions_list = np.array(predictions_list)
        labels_list = np.array(labels_list)
        logits_list = np.array(logits_list)

    ###################################################
    ##                   Accuracy                    ##
    ###################################################
    acc = accuracy_score(labels_list,predictions_list)
    print("Accuracy = {}".format(acc))

    '''
    ###################################################
    ##              precision-recall plot            ##
    ###################################################
    precision = dict()
    recall = dict()
    average_precision = dict()
    labels_reformed_list = []
    for i in range(24):
        label_reformed = (labels_list==i).astype(np.float32)
        precision[i], recall[i], _ = precision_recall_curve(label_reformed,logits_list[:,i])
        average_precision[i] = average_precision_score(label_reformed,logits_list[:,i])
        labels_reformed_list.append(label_reformed)
    labels_reformed_list = np.array(labels_reformed_list).transpose([1,0])

    precision["micro"], recall["micro"], _ = precision_recall_curve(labels_reformed_list.ravel(), logits_list.ravel())
    average_precision["micro"] = average_precision_score(labels_reformed_list, logits_list, average="micro")

    print("Average precision score, micro-averaged over all classes: {0:0.2f}".format(average_precision["micro"]))

    plt.figure()
    plt.step(recall["micro"], precision["micro"], color='b', alpha=0.2,
             where="post")
    plt.fill_between(recall["micro"], precision["micro"], step="post",
                     alpha=0.2, color='b')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.title("Average precision score, micro-averaged over all classes: AUC={0:0.2f}".format(average_precision["micro"]))
    plt.savefig("/home/johnson/Desktop/CEDL/homework1/precision_recall.jpg")

    ###############################################
    ##              confusion matrix             ##
    ###############################################
    cnf_matrix = confusion_matrix(labels_list, predictions_list)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, class=24, normalize=True,
                          title="Confusion matrix, with normalization")
    '''

if __name__ == '__main__':
  tf.app.run()
