import vgg_preprocessing
import inception_preprocessing
import tensorflow.contrib.slim as slim
import inception_resnet_v2 as iResnetV2
import tensorflow as tf
import os
import numpy as np


TRAIN_IMAGE = 'train_image.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_IMAGE = 'test_image.txt'
TEST_LABEL = 'test_label.txt'
PRETRAINED_MODEL = 'inception_resnet_v2_2016_08_30.ckpt'
MODEL_PATH = 'Model/iResnetV2_model.ckpt'

TOTAL_EPOCH = 31
BATCH_SIZE = 32
NUM_CLASSES = 24

#load file
def load_file(data_dir, data_type):
	'''
	load path from text file.	
	data_type: image or label
	'''
	print('Reading ' + data_type + '...')
	data = []	
	file = open(data_dir, 'r')
	
	for line in file:
		line = line.rstrip()
		if data_type == 'image':		
			line = os.path.join(line)			
			data.append(line)
		elif data_type == 'label':
			temp_data = np.load(line)
			data.append(temp_data)
	
	if data_type == 'label':
		data = np.concatenate(data)	
	
	print( 'Size: {}'.format(len(data)))
	
	return data

def load_preprocess(images_path, labels, is_training):
	'''
	Images preprocessing. Resize and crop the images
	'''
	no_processsed = tf.read_file(images_path)
	image = tf.image.decode_png(no_processsed, channels=3)
	image = tf.cast(image, tf.float32)
	pre_processed = tf.cond(is_training,
					true_fn=lambda: vgg_preprocessing.preprocess_image(image, 224, 224, is_training=False),
					false_fn=lambda: vgg_preprocessing.preprocess_image(image, 224, 224, is_training=False))
	return pre_processed, labels

def evaluate(sess, loss, correct_prediction, data_initial, feed_dict):
	'''
	Calculate loss and accuracy.
	'''
	sess.run(data_initial, feed_dict=feed_dict)

	data_loss = 0
	num_correct = 0
	num_samples = 0

	while True:
		try:
			_loss, _correct_prediction = sess.run([loss, correct_prediction], feed_dict={is_training: False})
			data_loss += _loss
			num_correct += _correct_prediction.sum() 
			num_samples += _correct_prediction.shape[0]
		except tf.errors.OutOfRangeError:
			break

	data_loss = data_loss / num_samples
	acc = num_correct / num_samples

	return data_loss, acc

def main(_):

	graph = tf.Graph()
	#Construct the network
	with graph.as_default():
		#Input and output
		is_training = tf.placeholder(dtype = tf.bool, name = 'is_training')
		images = tf.placeholder(dtype = tf.string, shape=(None,), name = 'images')
		labels = tf.placeholder(dtype = tf.int32, shape=(None,), name = 'labels')
	  
		#Data batch
		data = tf.contrib.data.Dataset.from_tensor_slices((images, labels))
		data = data.map(lambda image, label: load_preprocess(image, label, is_training))
		data = data.shuffle(buffer_size=10000)
		batched_data = data.batch(BATCH_SIZE)
		iterator = tf.contrib.data.Iterator.from_structure(batched_data.output_types,
                                                       batched_data.output_shapes)
		batched_images, batched_labels = iterator.get_next()
		data_initial = iterator.make_initializer(batched_data)
		
		#load pre-trained model
		with slim.arg_scope(iResnetV2.inception_resnet_v2_arg_scope()):
			logits, _ = iResnetV2.inception_resnet_v2(batched_images, num_classes = NUM_CLASSES, is_training = is_training)
		
		variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits'])
		init_fn = tf.contrib.framework.assign_from_checkpoint_fn(PRETRAINED_MODEL, variables_to_restore)
		
		#Define loss and optimizer
		tf.losses.sparse_softmax_cross_entropy(labels=batched_labels, logits=logits)
		loss = tf.losses.get_total_loss()		
		optimizer = tf.train.AdamOptimizer()
		train_op = optimizer.minimize(loss) 
		
		#Predict and calulate accuracy
		prediction = tf.to_int32(tf.argmax(logits, 1))
		correct_prediction = tf.equal(prediction, batched_labels)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
     
     
	#loading data   
	training_images = load_file(TRAIN_IMAGE, 'image')
	training_labels = load_file(TRAIN_LABEL, 'label')
	testing_images = load_file(TEST_IMAGE, 'image')
	testing_labels = load_file(TEST_IMAGE, 'label')
  
	#initialize
	sess = tf.Session(graph=graph)
	sess.run(init) 
	#saver.restore(sess, MODEL_PATH)
	init_fn(sess)
	
	training_log = open('log.txt', 'w')
	current_train_acc = 0 
	current_test_acc = 0	

	#Training
	for epoch in range(TOTAL_EPOCH):
		sess.run(data_initial, feed_dict={images: training_images,
											labels: training_labels,
											is_training: True})
		count = 0
		while True:
			try:
				_ = sess.run(train_op, feed_dict={is_training: True})
				print('[Epoch]: {} |[Batch]: {} |[accuracy] train: {} | test: {}'.format(epoch, count, current_train_acc, current_test_acc))
				count = count+1 
			except tf.errors.OutOfRangeError:
				break

		train_loss, train_acc = evaluate(sess, loss, correct_prediction, data_initial,
										feed_dict={images: training_images,
													labels: training_labels,
													is_training: True})
		print('[Epoch]: {} |[Train] loss: {} | accuracy: {}'.format(epoch, train_loss, train_acc))
		print('[Epoch]: {} |[Train] loss: {} | accuracy: {}'.format(epoch, train_loss, train_acc), file=training_log)

		#Save the current model
		if epoch%10 == 0:
			save_path = saver.save(sess, MODEL_PATH)
			print("Model updated and saved in file: %s" % save_path)

		#Testing
		if epoch%5 == 0:
			test_loss, test_acc = evaluate(sess, loss, correct_prediction, data_initial,
										feed_dict={images: testing_images,
													labels: testing_labels,
													is_training: False})
			print('\n[Epoch]: {} |[Test] loss: {} | accuracy: {}\n\n\n'.format(epoch, test_loss, test_acc))
			print('\n[Epoch]: {} |[Test] loss: {} | accuracy: {}\n\n\n'.format(epoch, test_loss, test_acc), file=training_log)
			
			current_test_acc = test_acc
		
		current_train_acc = train_acc

if __name__ == '__main__':
	tf.app.run()
