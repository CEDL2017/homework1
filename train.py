import os
import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from vgg import VGGNet
from datagenerator import ImageDataGenerator

from datetime import datetime
from tensorflow.contrib.data import Iterator

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of Epochs')
tf.app.flags.DEFINE_float('dropout_rate', 0.5, 'Dropout Rate')
tf.app.flags.DEFINE_integer('display_step', 20, 'Display Step')
tf.app.flags.DEFINE_integer('num_classes', 24, 'Number of Classes')
tf.app.flags.DEFINE_boolean('usingVGG', False, 'Set to True to use VGGnet model')

# train_layers = ['fc8', 'fc7', 'fc6']
train_layers = ['fc8', 'fc7', 'fc6']

train_file = "./dataset/train_list.txt"
val_file = "./dataset/test_list.txt"
filewriter_path = "./tensorboard"
checkpoint_path = "./checkpoints"

# Place data loading and preprocessing on the cpu

tr_data = ImageDataGenerator(train_file,
                             mode='training',
                             batch_size=FLAGS.batch_size,
                             num_classes=FLAGS.num_classes,
                             shuffle=True)
val_data = ImageDataGenerator(val_file,
                              mode='validation',
                              batch_size=FLAGS.batch_size,
                              num_classes=FLAGS.num_classes,
                              shuffle=False)

# create an reinitializable iterator given the dataset structure
iterator = Iterator.from_structure(tr_data.data.output_types,
                                   tr_data.data.output_shapes)
next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x_hand = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
x_head = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
if FLAGS.usingVGG == True:
    model = VGGNet(x_hand, x_head, keep_prob, FLAGS.num_classes, train_layers)
else:
    model = AlexNet(x_hand, x_head, keep_prob, FLAGS.num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / FLAGS.batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / FLAGS.batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights_two_stream(sess)

    print("{}: Start training.".format(datetime.now()))
    print("{}: Tensorboard path at --logdir {}".format(datetime.now(), filewriter_path))

    # Loop over number of epochs
    for epoch in range(FLAGS.num_epochs):

        print("{}: Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch_hand, img_batch_head, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x_hand: img_batch_hand,
                                          x_head: img_batch_head,
                                          y: label_batch,
                                          keep_prob: FLAGS.dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % FLAGS.display_step == 0:
                s = sess.run(merged_summary, feed_dict={x_hand: img_batch_hand,
                                                        x_head: img_batch_head,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        #Validate the model on the entire validation set

        print("{}: Start Testing".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        predictions = []
        for _ in range(val_batches_per_epoch):

            img_batch_hand, img_batch_head, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x_hand: img_batch_hand,
                                                x_head: img_batch_head,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
            sco = sess.run(score, feed_dict={x_hand: img_batch_hand,
                                                x_head: img_batch_head,
                                                y: label_batch,
                                                keep_prob: 1.})
            sco = tf.argmax(sco, 1)
            predictions = np.concatenate([predictions, sco.eval()])
        np.save('predictions.npy', predictions)
        test_acc /= test_count
        print("{}: Testing Accuracy = {:.4f}".format(datetime.now(), test_acc))
        print("{}: Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{}: Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
