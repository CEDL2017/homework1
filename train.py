from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import time

from input import ImageReader
from model import ResNet50

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
DATA_LIST = './list/train_list.txt'
BATCH_SIZE = 64
RANDOM_SCALE = False
RANDOM_MIRROR = True
NUM_CLASSES = 24
NUM_STEPS = 60001
NOT_RESTORE_LAST = False
SNAPSHOT_DIR = './snapshots'

train_layers = ['fc24']

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)

   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    coord = tf.train.Coordinator()
    #restore_from_npy()
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            DATA_LIST,
            [1080, 1920],
            RANDOM_SCALE,
            RANDOM_MIRROR,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(BATCH_SIZE)

    net = ResNet50({'data': image_batch}, is_training=False, num_classes=NUM_CLASSES)

    restore_var = [v for v in tf.global_variables() if 'fc24' not in v.name or not NOT_RESTORE_LAST]
    trainable = [v for v in tf.trainable_variables() if 'fc24' in v.name] # Fine-tune only the last layers.

    fc_out = net.layers['fc24']
    gt = tf.one_hot(indices=label_batch, depth=NUM_CLASSES)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc_out, labels=gt)
    reduced_loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=5e-5)
    train_op = optimizer.minimize(reduced_loss, var_list=trainable)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    # Load variables if the checkpoint is provided.
    """
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, CKPT_PATH)

    """
    ckpt = tf.train.get_checkpoint_state(SNAPSHOT_DIR)

    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        load_step = 0
    

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for step in range(NUM_STEPS):
        start_time = time.time()

        if step % 50 == 0:
            loss_value, _ = sess.run([reduced_loss, train_op])
            save(saver, sess, SNAPSHOT_DIR, step)
        else:
            loss_value, _ = sess.run([reduced_loss, train_op])

        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
