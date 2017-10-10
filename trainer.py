import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time
import subprocess

slim = tf.contrib.slim

# prepare images total number

place = ['house', 'lab', 'office']
# place = ['house']
num = ['1', '2', '3']
# num = num[0:1]
kind = ['Lhand', 'Rhand']
# kind = kind[0:1]
img_num = 0

for p in place:
    for n in num:
        for k in kind:
            path = '../frames/train/' + p + '/' + n + '/' + k
            png_list = os.listdir(path)
            img_num += len(png_list)
            for pic in png_list:
                npic = pic.replace('Image', '')
                npic = npic.replace('.png', '')
                if int(npic) < 100:
                    npic = 'Image' + npic.zfill(3) + '.png'
                    os.rename(path + '/' + pic, path + '/' + npic)

for k in kind:
    path = '../frames/train/' + 'lab' + '/' + '4' + '/' + k
    png_list = os.listdir(path)
    img_num += len(png_list)
    for pic in png_list:
        npic = pic.replace('Image', '')
        npic = npic.replace('.png', '')
        if int(npic) < 100:
            npic = 'Image' + npic.zfill(3) + '.png'
            os.rename(path + '/' + pic, path + '/' + npic)

print('number of images:', img_num)

# loading labels data 

place = ['house', 'lab', 'office']
num = ['1', '2', '3']
# num = num[0:1]
kind = ['left', 'right']
# kind = kind[0:1]

path = '../labels/'

label = np.array([])

for p in place:
    label_list = os.listdir(path + p)
    for l in label_list:
        label_name = l
        if 'obj' in label_name: 
            label_name = label_name.replace('.npy', '')
            for k in kind:
                if k in label_name: 
                    if p == 'lab':
                        for n in ['1', '2', '3', '4']:
                            if n in label_name:
                                print(p + '/' + l)
                                if len(label) == 0:
                                    label = np.load(path + p + '/' + l)
                                else:
                                    label = np.concatenate([label, np.load(path + p + '/' + l)])
                    else:
                        for n in num:
                            if n in label_name:
                                print(p + '/' + l)
                                if len(label) == 0:
                                    label = np.load(path + p + '/' + l)
                                else:
                                    label = np.concatenate([label, np.load(path + p + '/' + l)])

label = label.astype(int)
print('label data loaded, and the number of labels is:', len(label))

#State where your log file is at. If it doesn't exist, create it.
log_dir = '../log'

#State where your checkpoint file is
checkpoint_file = '../inception_resnet_v2_2016_08_30.ckpt'

#State the number of classes to predict:
obj = { 'free':0,
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

num_classes = len(obj)
num_epochs = 30
batch_size = 16
initial_learning_rate = 0.01
learning_rate_decay_factor = 0.9
num_epochs_before_decay = 2

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)

with tf.Graph().as_default() as graph:
    tf.logging.set_verbosity(tf.logging.INFO)

    num_batches_per_epoch = int(img_num / batch_size)
    print("num_batches_per_epoch:", num_batches_per_epoch)
    num_steps_per_epoch = num_batches_per_epoch
    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

    # prepare image FIFOQueue
    height = 299
    width = 299
    num_threads = 4

    image_names = tf.train.match_filenames_once('../frames/train/*/*/*hand/Image*.png')
    image_queue = tf.train.string_input_producer(image_names, shuffle = False)

    image_reader = tf.WholeFileReader()
    _, image_value = image_reader.read(image_queue)

    image_tf = tf.image.decode_png(image_value, channels = 3)
    image_tf = tf.image.resize_images(image_tf, [height, width])
    image_tf.set_shape((height, width, 3))
    # image_tf = 2 * (image_tf / 255.0) - 1.0
    image_tf = tf.subtract(tf.multiply(2.0, tf.divide(image_tf, 255.0)), 1.0)
    # image_tf = inception_preprocessing.preprocess_image(image_tf, height, width, is_training = True)

    # prepare label FIFOQueue
    label_tf = tf.convert_to_tensor(label, dtype = tf.int32)
    label_queue = tf.train.slice_input_producer([label_tf], shuffle = False)

    # creating a batch of images and labels
    batch_image, batch_label = tf.train.batch([[image_tf], label_queue], 
                                            batch_size = batch_size, 
                                            num_threads = num_threads,
                                            capacity = num_threads * batch_size,
                                            enqueue_many = True,
                                            allow_smaller_final_batch = True)

    #Create the model inference
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(batch_image, 
                                                 num_classes = num_classes, 
                                                 is_training = True)

    #Define the scopes that you want to exclude for restoration
    exclude = ['InceptionResnetV2/AuxLogits',
               'InceptionResnetV2/Logits']
    variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
    one_hot_labels = slim.one_hot_encoding(batch_label, 
                                           num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, 
                                           logits = logits)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    all_losses = [loss] + regularization_losses
    total_loss = tf.add_n(all_losses, name='total_loss')
    # total_loss = tf.losses.get_total_loss()

    global_step = get_or_create_global_step()

    lr = tf.train.exponential_decay(
        learning_rate = initial_learning_rate,
        global_step = global_step,
        decay_steps = decay_steps,
        decay_rate = learning_rate_decay_factor,
        staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)

    # get variables to be trained
    variables_to_train = []
    with tf.Session() as sess:
        for v in tf.trainable_variables():
            variables_to_train += [v]
    variables_to_train = variables_to_train[len(variables_to_train) - 4 : len(variables_to_train)]

    train_op = slim.learning.create_train_op(total_loss = total_loss, 
                                             optimizer = optimizer,
                                             variables_to_train = variables_to_train)

    predictions = tf.argmax(end_points['Predictions'], 1)
    probabilities = end_points['Predictions']
    accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, batch_label)
    metrics_op = tf.group(accuracy_update, probabilities)
    
    print('Graph builded')
    
    #Now finally create all the summaries you need to monitor and group them into one summary op.
    tf.summary.scalar('losses/Total_Loss', total_loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('learning_rate', lr)
    my_summary_op = tf.summary.merge_all()

    #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
    def train_step(sess, train_op, global_step):
        total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
        return total_loss, global_step_count

    #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
    saver = tf.train.Saver(variables_to_restore)
    #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
    sv = tf.train.Supervisor(logdir = log_dir, summary_op = None)
    print('Ready to run the session')

    #Run the managed session
    with sv.managed_session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        saver.restore(sess, '../inception_resnet_v2_2016_08_30.ckpt')

        for step in range(num_steps_per_epoch * num_epochs):

            if step % 10 == 0:
                loss, _ = train_step(sess, train_op, sv.global_step)
                summaries = sess.run(my_summary_op)
                sv.summary_computed(sess, summaries)

            else:
                loss, _ = train_step(sess, train_op, sv.global_step)

            if step % num_batches_per_epoch == 0:
                logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                learning_rate_value, accuracy_value = sess.run([lr, accuracy])

                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming Accuracy: %s', accuracy_value)

                predictions_value, labels_value = sess.run([predictions, batch_label])
                
                print('predictions: \n', predictions_value)
                print('Labels: \n', labels_value)

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

        #We log the final training loss and accuracy
        logging.info('Final Loss: %s', loss)
        logging.info('Final Accuracy: %s', sess.run(accuracy))

        #Once all the training has been done, save the log files and checkpoint model
        logging.info('Finished training! Saving model to disk now.')
        # saver.save(sess, "../model.ckpt")
        sv.saver.save(sess, sv.save_path, global_step = sv.global_step)