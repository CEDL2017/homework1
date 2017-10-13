from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
# for pr_point
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

# for pr_point
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/metrics/python/ops/metric_ops.py
def _remove_squeezable_dimensions(predictions, labels, weights):
  labels, predictions = confusion_matrix.remove_squeezable_dimensions(
      labels, predictions)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  if weights is not None:
    weights = ops.convert_to_tensor(weights)
    predictions_shape = predictions.get_shape()
    predictions_rank = predictions_shape.ndims
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims

    if (predictions_rank is not None) and (weights_rank is not None):
      # Use static rank.
      if weights_rank - predictions_rank == 1:
        weights = array_ops.squeeze(weights, [-1])
    elif (weights_rank is None) or (
        weights_shape.dims[-1].is_compatible_with(1)):
      # Use dynamic rank
      weights = control_flow_ops.cond(
          math_ops.equal(array_ops.rank(weights),
                         math_ops.add(array_ops.rank(predictions), 1)),
          lambda: array_ops.squeeze(weights, [-1]),
          lambda: weights)
  return predictions, labels, weights
def _streaming_confusion_matrix_at_thresholds(
    predictions, labels, thresholds, weights=None, includes=None):
  all_includes = ('tp', 'fn', 'tn', 'fp')
  if includes is None:
    includes = all_includes
  else:
    for include in includes:
      if include not in all_includes:
        raise ValueError('Invaild key: %s.' % include)

  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions, labels, weights)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  num_thresholds = len(thresholds)

  # Reshape predictions and labels.
  predictions_2d = array_ops.reshape(predictions, [-1, 1])
  labels_2d = array_ops.reshape(
      math_ops.cast(labels, dtype=tf.bool), [-1, 1])

  # Use static shape if known.
  num_predictions = predictions_2d.get_shape().as_list()[0]

  # Otherwise use dynamic shape.
  if num_predictions is None:
    num_predictions = array_ops.shape(predictions_2d)[0]
  thresh_tiled = array_ops.tile(
      array_ops.expand_dims(array_ops.constant(thresholds), [1]),
      array_ops.stack([1, num_predictions]))

  # Tile the predictions after thresholding them across different thresholds.
  pred_is_pos = math_ops.greater(
      array_ops.tile(array_ops.transpose(predictions_2d), [num_thresholds, 1]),
      thresh_tiled)
  if ('fn' in includes) or ('tn' in includes):
    pred_is_neg = math_ops.logical_not(pred_is_pos)

  # Tile labels by number of thresholds
  label_is_pos = array_ops.tile(array_ops.transpose(labels_2d), [num_thresholds, 1])
  if ('fp' in includes) or ('tn' in includes):
    label_is_neg = math_ops.logical_not(label_is_pos)

  if weights is not None:
    broadcast_weights = weights_broadcast_ops.broadcast_weights(
        math_ops.to_float(weights), predictions)
    weights_tiled = array_ops.tile(array_ops.reshape(
        broadcast_weights, [1, -1]), [num_thresholds, 1])
    thresh_tiled.get_shape().assert_is_compatible_with(
        weights_tiled.get_shape())
  else:
    weights_tiled = None

  values = {}
  update_ops = {}

  if 'tp' in includes:
    true_positives = _create_local('true_positives', shape=[num_thresholds])
    is_true_positive = math_ops.to_float(
        math_ops.logical_and(label_is_pos, pred_is_pos))
    if weights_tiled is not None:
      is_true_positive *= weights_tiled
    update_ops['tp'] = state_ops.assign_add(
        true_positives, math_ops.reduce_sum(is_true_positive, 1))
    values['tp'] = true_positives

  if 'fn' in includes:
    false_negatives = _create_local('false_negatives', shape=[num_thresholds])
    is_false_negative = math_ops.to_float(
        math_ops.logical_and(label_is_pos, pred_is_neg))
    if weights_tiled is not None:
      is_false_negative *= weights_tiled
    update_ops['fn'] = state_ops.assign_add(
        false_negatives, math_ops.reduce_sum(is_false_negative, 1))
    values['fn'] = false_negatives

  if 'tn' in includes:
    true_negatives = _create_local('true_negatives', shape=[num_thresholds])
    is_true_negative = math_ops.to_float(
        math_ops.logical_and(label_is_neg, pred_is_neg))
    if weights_tiled is not None:
      is_true_negative *= weights_tiled
    update_ops['tn'] = state_ops.assign_add(
        true_negatives, math_ops.reduce_sum(is_true_negative, 1))
    values['tn'] = true_negatives

  if 'fp' in includes:
    false_positives = _create_local('false_positives', shape=[num_thresholds])
    is_false_positive = math_ops.to_float(
        math_ops.logical_and(label_is_neg, pred_is_pos))
    if weights_tiled is not None:
      is_false_positive *= weights_tiled
    update_ops['fp'] = state_ops.assign_add(
        false_positives, math_ops.reduce_sum(is_false_positive, 1))
    values['fp'] = false_positives

  return values, update_ops

def get_streaming_curve_points(labels=None,
                               predictions=None,
                               weights=None,
                               num_thresholds=200,
                               metrics_collections=None,
                               updates_collections=None,
                               curve='ROC',
                               name=None):
  with variable_scope.variable_scope(name, 'curve_points', (labels, predictions,
                                                            weights)):
  # with tf.name_scope("eval"):
    if curve != 'ROC' and curve != 'PR':
      raise ValueError('curve must be either ROC or PR, %s unknown' % (curve))
    kepsilon = 1e-7  # to account for floating point imprecisions
    thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                  for i in range(num_thresholds - 2)]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

    values, update_ops = _streaming_confusion_matrix_at_thresholds(
        labels=labels,
        predictions=predictions,
        thresholds=thresholds,
        weights=weights)

    # Add epsilons to avoid dividing by 0.
    epsilon = 1.0e-6

    def compute_points(tp, fn, tn, fp):
      """Computes the roc-auc or pr-auc based on confusion counts."""
      rec = math_ops.div(tp + epsilon, tp + fn + epsilon)
      if curve == 'ROC':
        fp_rate = math_ops.div(fp, fp + tn + epsilon)
        return fp_rate, rec
      else:  # curve == 'PR'.
        prec = math_ops.div(tp + epsilon, tp + fp + epsilon)
        return rec, prec

    xs, ys = compute_points(values['tp'], values['fn'], values['tn'],
                            values['fp'])
    points = array_ops.stack([xs, ys], axis=1)
    xs2, ys2 = compute_points(update_ops['tp'], update_ops['fn'], update_ops['tn'],
                            update_ops['fp'])
    update_op = array_ops.stack([xs2, ys2], axis=1)

    if metrics_collections:
      ops.add_to_collections(metrics_collections, points)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return points, update_op
  
# for confusion_matrix
def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=tf.float32):
    """Creates a new local variable.
    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      collections: A list of collection names to which the Variable will be added.
      validate_shape: Whether to validate the shape of the variable.
      dtype: Data type of the variables.
    Returns:
      The created variable.
    """
    # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
    collections = list(collections or [])
    collections += [tf.GraphKeys.LOCAL_VARIABLES]
    return tf.Variable(
        initial_value=tf.zeros(shape, dtype=dtype),
        name=name,
        trainable=False,
        collections=collections,
        validate_shape=validate_shape)

def get_streaming_metrics(label, prediction, num_classes):
    with tf.name_scope("eval"):
        batch_confusion = tf.confusion_matrix(label, prediction,
                                              num_classes=num_classes,
                                              name='batch_confusion')

        confusion = _create_local('confusion_matrix',
                                  shape=[num_classes, num_classes],
                                  dtype=tf.int32)
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion.assign(confusion + batch_confusion)
        # Cast counts to float so tf.summary.image renormalizes to [0,255]
        confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
                                     [1, num_classes, num_classes, 1])

    return confusion, confusion_update
