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

# place = ['house', 'lab', 'office']
place = ['house']
# num = ['1', '2', '3']
num = ['1']
# kind = ['Lhand', 'Rhand']
kind = ['Lhand']
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

# for k in kind:
#     path = '../frames/train/' + 'lab' + '/' + '4' + '/' + k
#     png_list = os.listdir(path)
#     img_num += len(png_list)
#     for pic in png_list:
#         npic = pic.replace('Image', '')
#         npic = npic.replace('.png', '')
#         if int(npic) < 100:
#             npic = 'Image' + npic.zfill(3) + '.png'
#             os.rename(path + '/' + pic, path + '/' + npic)


print('count number of images:', img_num)

# place = ['house', 'lab', 'office']
# num = ['1', '2', '3']
num = ['1']
# kind = ['left', 'right']
kind = ['left']

path = '../labels/'

label = np.array([])

for p in place:
    lab_list = os.listdir(path + p)
    for lab in lab_list:
        ali = lab
        if 'obj' in ali:
            ali = ali.replace('.npy', '')
            for k in kind:
                if k in ali:
                    for n in num:
                        if n in ali:
                            if len(label) == 0:
                                label = np.load(path + p + '/' + lab)
                            else:
                                label = np.concatenate([label, np.load(path + p + '/' + lab)])

print('label data loaded, and the data size is:', len(label))

# prepare image data
img_names = tf.train.match_filenames_once('../frames/train/house/1/Lhand/Image*.png')
img_queue = tf.train.string_input_producer(img_names)

img_reader = tf.WholeFileReader()
_, img_value = img_reader.read(img_queue)

raw_img = tf.image.decode_png(img_value, channels = 3)
raw_img = tf.image.resize_images(raw_img, [300, 300])
raw_img.set_shape((300, 300, 3))

image = []
init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
#     print(sess.run(img_names))
    img_num = 2
    for i in range(img_num):
        image_tensor = sess.run([raw_img])   
        image += [image_tensor[0]]
#         print(image_tensor[0].shape)
    
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    
print('image data loaded, and the data size is: ', len(image))

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


#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 1

#State your batch size
batch_size = 8

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2


def load_batch(raw_image, label, batch_size, height = 300, width = 300, is_training=True):
    #Perform the correct preprocessing for this image depending if it is training or evaluating
    print('start preprocess')
    image = inception_preprocessing.preprocess_image(tf.convert_to_tensor(raw_image),
                                                     height, 
                                                     width, 
                                                     is_training)
    print('1')
    time.sleep(2)
    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    print('2')
    time.sleep(2)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    print('3')
    time.sleep(2)
    raw_image = tf.squeeze(raw_image)
    print('4')
    time.sleep(2)
    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)
    
    print('5')
    time.sleep(2)

    return images, raw_images, labels


if not os.path.exists(log_dir):
    os.mkdir(log_dir)

with tf.Graph().as_default() as graph:
    tf.logging.set_verbosity(tf.logging.INFO)

    im, im1 = tf.train.batch(
        tensors = [tf.convert_to_tensor(image), 
                   tf.convert_to_tensor(image)],
        batch_size = batch_size,
        enqueue_many = True,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    # im, _, lb = load_batch(raw_image = image, label = label, batch_size = batch_size)
    num_batches_per_epoch = int(img_num / batch_size)
    num_steps_per_epoch = num_batches_per_epoch
    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

    print('here, wait 5 secs')
    time.sleep(5)

    #Create the model inference
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(im, 
                                                 num_classes = num_classes, 
                                                 is_training = True)

    print('Done')
    time.sleep(5)

    #Define the scopes that you want to exclude for restoration
    exclude = ['InceptionResnetV2/Logits', 
               'InceptionResnetV2/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

    one_hot_labels = slim.one_hot_encoding(lb, 
                                           num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, 
                                           logits = logits)
    total_loss = tf.losses.get_total_loss()

    global_step = get_or_create_global_step()

    lr = tf.train.exponential_decay(
        learning_rate = initial_learning_rate,
        global_step = global_step,
        decay_steps = decay_steps,
        decay_rate = learning_rate_decay_factor,
        staircase = True)

    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    predictions = tf.argmax(end_points['Predictions'], 1)
    probabilities = end_points['Predictions']
    accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, lb)
    metrics_op = tf.group(accuracy_update, probabilities)
    
    print('Stage 1')
    
    #Now finally create all the summaries you need to monitor and group them into one summary op.
    tf.summary.scalar('losses/Total_Loss', total_loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('learning_rate', lr)
    my_summary_op = tf.summary.merge_all()

    #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
    def train_step(sess, train_op, global_step):
        #Check the time for each sess run
        start_time = time.time()
        total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
        time_elapsed = time.time() - start_time

        logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

        return total_loss, global_step_count

    #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint_file)

    #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
    sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)
    
    print('Stage 2')

    #Run the managed session
    with sv.managed_session() as sess:
        for step in xrange(num_steps_per_epoch * num_epochs):
            if step % num_batches_per_epoch == 0:
                logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming Accuracy: %s', accuracy_value)

                # optionally, print your logits and predictions for a sanity check that things are going fine.
                logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, lb])
                print('logits: \n', logits_value)
                print('Probabilities: \n', probabilities_value)
                print('predictions: \n', predictions_value)
                print('Labels: \n', labels_value)

            #Log the summaries every 10 step.
            if step % 10 == 0:
                loss, _ = train_step(sess, train_op, sv.global_step)
                summaries = sess.run(my_summary_op)
                sv.summary_computed(sess, summaries)

            #If not, simply run the training step
            else:
                loss, _ = train_step(sess, train_op, sv.global_step)

        #We log the final training loss and accuracy
        logging.info('Final Loss: %s', loss)
        logging.info('Final Accuracy: %s', sess.run(accuracy))

        #Once all the training has been done, save the log files and checkpoint model
        logging.info('Finished training! Saving model to disk now.')
        saver.save(sess, "../model.ckpt")
        sv.saver.save(sess, sv.save_path, global_step = sv.global_step)