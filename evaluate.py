import numpy as np
import tensorflow as tf
import os
import time

from input import ImageReader
from model import ResNet50

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
DATA_LIST = './list/test_list.txt'
NUM_CLASSES = 24
NUM_STEPS = 6388
SNAPSHOT_DIR = './snapshots'

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    coord = tf.train.Coordinator()

    tf.reset_default_graph()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            DATA_LIST,
            [1080, 1920], # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    net = ResNet50({'data': image_batch}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    fc_out = net.layers['fc24']
    softmax = tf.nn.softmax(fc_out)
    pred = tf.argmax(softmax, axis=-1)

    # convert label to onehot
    gt = tf.one_hot(indices=label_batch, depth=NUM_CLASSES)

    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, label_batch, num_classes=NUM_CLASSES)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    sess.run(global_init)
    sess.run(local_init)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)

    ckpt = tf.train.get_checkpoint_state(SNAPSHOT_DIR)

    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        load_step = 0

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    true_ct = 0
    for step in range(NUM_STEPS):
        _pred, _label, _cm = sess.run([pred, label_batch, update_op])

        if _pred == _label:
            true_ct += 1
        if step % 100 == 0:
            print('Finish {0}/{1}'.format(step, NUM_STEPS))

    print('correct: {}'.format(true_ct))
    print('acc: {}'.format((float)(true_ct)/NUM_STEPS))
    np.save('./results/cm.npy', _cm)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
