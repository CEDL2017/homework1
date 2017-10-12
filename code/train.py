import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import vgg_preprocessing
import inception_preprocessing
# from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np


train_data_list_dir = 'train_list.txt'
train_label_list_dir = 'train_label_list.txt'
test_data_list_dir = 'test_list.txt'
test_label_list_dir = 'test_label_list.txt'



total_epoch = 31
batch_size = 32
learning_rate = 0.001

# X = tf.placeholder(tf.float32, (None, 244, 244, 3))
# Y = tf.placeholder(tf.float32, (None, 1))


# load label and image paths 
def load_input(data_dir, label_dir):
	file_names = []
	with open(data_dir, 'r') as f:
		for line in f:
			line = line.rstrip()
			line = os.path.join(line)
			file_names.append(line)
	print('read in file name list')
	print( 'number of image paths: {}'.format(len(file_names)))

	labels = []
	with open(label_dir, 'r') as f:
		for line in f:
			line = line.rstrip()
			tmp_data = np.load(line)
			labels.append(tmp_data)
	labels = np.concatenate(labels)
	print('read in labels')
	print('number of labels: {}'.format(len(labels)))

	return file_names, labels

#load images and preprocess the images with vgg16 preprocessing andinception preprocessing and compare two results
def load_image(image_path, label, is_training):
	image_buffer = tf.read_file(image_path)
	image = tf.image.decode_png(image_buffer, channels=3)
	image = tf.cast(image, tf.float32)
	image = tf.cond(is_training,
					true_fn=lambda: inception_preprocessing.preprocess_image(image, 224, 224, is_training=True),
					false_fn=lambda: inception_preprocessing.preprocess_image(image, 224, 224, is_training=False))
	return image, label



graph = tf.Graph()
with graph.as_default():

	# initialize the placeholders
	is_training = tf.placeholder(dtype = tf.bool)
	file_names = tf.placeholder(dtype = tf.string, shape=(None,))
	labels = tf.placeholder(dtype = tf.int32, shape=(None,))

	# prepocessing data and set them into batches
	sliced_data = tf.contrib.data.Dataset.from_tensor_slices((file_names, labels))
	data = sliced_data.map(lambda file_name, label: load_image(file_name, label, is_training))
	data = data.shuffle(buffer_size=10000)
	batched_data = data.batch(batch_size)
	iterator = tf.contrib.data.Iterator.from_structure(batched_data.output_types,
                                                       batched_data.output_shapes)
	batch_images, batch_labels = iterator.get_next()
	dataset_initialize = iterator.make_initializer(batched_data)
	# create resnet
	# use inception resnet 
	with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(batch_images, num_classes = 24, is_training = True)
    # resize the output of inception_resnet to fit what we want 
	logits = tf.reshape(logits, [-1, 24])

	# exclude the weights of last layer
	variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits'])
	# restore the weight form pretrained model
	init_fn = tf.contrib.framework.assign_from_checkpoint_fn('inception_resnet_v2_2016_08_30.ckpt', variables_to_restore)

	# get total loss
	tf.losses.sparse_softmax_cross_entropy(labels=batch_labels, logits=logits)
	loss = tf.losses.get_total_loss()

	# using ADAM optimizer
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss)


	prediction = tf.to_int32(tf.argmax(logits, 1))
	correct_prediction = tf.equal(prediction, batch_labels)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	init = tf.global_variables_initializer()


def evaluate(sess, loss, correct_prediction, dataset_init_op, feed_dict):
	sess.run(dataset_init_op, feed_dict=feed_dict)

	total_loss = 0
	hit_number = 0
	total_sample = 0

	while True:
		try:
			# get correct predictions and 
			_loss, _correct_prediction = sess.run([loss, correct_prediction], feed_dict={is_training: False})

			total_loss += _loss
			hit_number += _correct_prediction.sum() 
			total_sample += _correct_prediction.shape[0]
		except tf.errors.OutOfRangeError:
			break

	data_loss = total_loss / total_sample
	accuracy = hit_number / total_sample

	return data_loss, accuracy


def main(_):
	# load file paths
	train_file_names, train_labels = load_input(train_data_list_dir, train_label_list_dir)
	test_file_names, test_labels = load_input(test_data_list_dir, test_label_list_dir)
	sess = tf.Session(graph=graph)
	# initialze variables
	sess.run(init)
	# load in pretrained weight
	init_fn(sess)
	# saver.restore(sess, 'inception_resnet_v2_2016_08_30.ckpt')
	training_log = open('training_changing_opt_log.txt', 'w')

	# run totally 30 epoches
	for epoch in range(total_epoch):
		print('\nnow in epoch: '+ str(epoch))

		# load in images
		sess.run(dataset_initialize, feed_dict={file_names: train_file_names,
											labels: train_labels,
											is_training: True})
		i = 0
		while True:
			try:
				print('data_batch: ' + str(i))
				i = i+1
				# training
				_ = sess.run(train_op, feed_dict={is_training: True})    
			except tf.errors.OutOfRangeError:
				break

		# evaluate the training accuracy of each epoch
		train_loss, train_acc = evaluate(sess, loss, correct_prediction, dataset_initialize,
										feed_dict={file_names: train_file_names,
													labels: train_labels,
													is_training: True})
		print('[Train] loss: {}, accuracy: {}'.format(train_loss, train_acc))
		print('[epoch]: {}, [Train] loss: {},  accuracy: {}'.format(epoch, train_loss, train_acc), file=training_log)

		# evaluate testing accuracy of every five epoches
		if(epoch%5 == 0):
			train_loss, train_acc = evaluate(sess, loss, correct_prediction, dataset_initialize,
										feed_dict={file_names: test_file_names,
													labels: test_labels,
													is_training: False})
			print('\n[test] loss: {}, accuracy: {}\n\n\n'.format(train_loss, train_acc))
			print('\n[epoch]: {}, [test] loss: {}, accuracy: {}\n\n\n'.format(epoch, train_loss, train_acc), file=training_log)


if __name__ == '__main__':
	tf.app.run()
