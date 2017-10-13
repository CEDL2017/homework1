import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from PIL import Image
import os
from time import time
import re
import cv2 as cv
import sys




for i in range(len(sys.argv)):
	if sys.argv[i].startswith('--o='):
		OUTPUT_DIR = sys.argv[1][4:] + '/'
	else:
		OUTPUT_DIR = './'

	if sys.argv[i].startswith('--LR='):
		LR = sys.argv[1][5:]
	else:
		LR = 10**(-3)

	if sys.argv[i].startswith('--epoch='):
		EPOCH= sys.argv[1][8:]
	else:
		EPOCH = 30


BATCH_SIZE = 48
INPUT_WIDTH = 96
INPUT_HEIGHT = int(INPUT_WIDTH * 9 / 16)

NUM_CLASS = 24

epoch_list=[]
accuracy_list=[]
loss_list=[]

# resnet_v2(inputs, blocks, num_classes=None, is_training=None, global_pool=True, output_stride=None, include_root_block=True, reuse=None, scope=None)
'''
def one_hot(inputs, num_class):
	# inputs = [int(a) for a in inputs]
	b = np.zeros((1, num_class))
	b[np.arange(1), inputs] = 1
	return b
'''

def resNet(inputs): # 2 + 6 + 8 layers

	with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn = tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
		net = slim.conv2d(inputs, 64 , [7, 7], stride = 2, scope = 'conv1')
		net = slim.max_pool2d(net, kernel_size = [3, 3], stride = 2, padding = 'SAME', scope = 'max_pool1')
		short_cut = net


		net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope = 'conv2_1')
		net = tf.add(net, short_cut)
		short_cut = net
		net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope = 'conv2_2')
		net = tf.add(net, short_cut)
		short_cut = net
		net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope = 'conv2_3')
		net = tf.add(net, short_cut)

		net = slim.conv2d(net, 128, [3, 3], stride = 2, scope = 'conv3_1')
		net = slim.conv2d(net, 128, [3, 3])
		short_cut = net
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope = 'conv3_2')
		net = tf.add(net, short_cut)
		short_cut = net
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope = 'conv3_3')
		net = tf.add(net, short_cut)
		short_cut = net
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope = 'conv3_4')
		net = tf.add(net, short_cut)
		
		net = slim.conv2d(net, 256, [3, 3], stride = 2, scope = 'conv4_1')
		net = slim.conv2d(net, 256, [3, 3])
		short_cut = net
		net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope = 'conv4_2')
		net = tf.add(net, short_cut)
		short_cut = net
		net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope = 'conv4_3')
		net = tf.add(net, short_cut)
		short_cut = net
		net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope = 'conv4_4')
		net = tf.add(net, short_cut)
		
		net = slim.avg_pool2d(net, kernel_size = [3, 3], padding = 'SAME')
		net = slim.flatten(net)
		logits = slim.fully_connected(inputs = net, num_outputs = NUM_CLASS, scope = 'fc')
		# prediction = tf.nn.softmax(logits, dim = -1)
	return logits




# for local
train_image_path = "./handcam/frames/train/"
test_image_path = "./handcam/frames/test/"
label_path = "./handcam/labels/"

'''
# for htc deepQ
data_env = os.environ['GRAPE_DATASET_DIR']

train_image_path = os.path.join(data_env,'frames/train/')
test_image_path = os.path.join(data_env,'frames/test/')
label_path = os.path.join(data_env,'labels/')



# for server
train_image_path = "/Disk2/cedl/handcam/frames/train/"
test_image_path = "/Disk2/cedl/handcam/frames/test/"
label_path = "/Disk2/cedl/handcam/labels/"
'''

train_image_file = ['house/1/Lhand', 'house/2/Lhand', 'house/3/Lhand', 
					'house/1/Rhand', 'house/2/Rhand', 'house/3/Rhand',
					'lab/1/Lhand', 'lab/2/Lhand', 'lab/3/Lhand', 'lab/4/Lhand',
					'lab/1/Rhand', 'lab/2/Rhand', 'lab/3/Rhand', 'lab/4/Rhand',
					'office/1/Lhand', 'office/2/Lhand', 'office/3/Lhand', 
					'office/1/Rhand', 'office/2/Rhand', 'office/3/Rhand',]

test_image_file = ['house/1/Lhand', 'house/2/Lhand', 'house/3/Lhand', 
					'house/1/Rhand', 'house/2/Rhand', 'house/3/Rhand',
					'lab/1/Lhand', 'lab/2/Lhand', 'lab/3/Lhand', 'lab/4/Lhand',
					'lab/1/Rhand', 'lab/2/Rhand', 'lab/3/Rhand', 'lab/4/Rhand',
					'office/1/Lhand', 'office/2/Lhand', 'office/3/Lhand', 
					'office/1/Rhand', 'office/2/Rhand', 'office/3/Rhand',]

train_label_file = ['house/obj_left1.npy', 'house/obj_left2.npy', 'house/obj_left3.npy',
					'house/obj_right1.npy', 'house/obj_right2.npy', 'house/obj_right3.npy',
	 				'lab/obj_left1.npy', 'lab/obj_left2.npy', 'lab/obj_left3.npy', 'lab/obj_left4.npy',
	 				'lab/obj_right1.npy', 'lab/obj_right2.npy', 'lab/obj_right3.npy', 'lab/obj_right4.npy',
 					'office/obj_left1.npy', 'office/obj_left2.npy', 'office/obj_left3.npy',
 					'office/obj_right1.npy', 'office/obj_right2.npy', 'office/obj_right3.npy']

test_label_file = ['house/obj_left4.npy', 'house/obj_left5.npy', 'house/obj_left6.npy',
					'house/obj_right4.npy', 'house/obj_right5.npy', 'house/obj_right6.npy',
	 				'lab/obj_left5.npy', 'lab/obj_left6.npy', 'lab/obj_left7.npy', 'lab/obj_left8.npy',
	 				'lab/obj_right5.npy', 'lab/obj_right6.npy', 'lab/obj_right7.npy', 'lab/obj_right8.npy',
 					'office/obj_left4.npy', 'office/obj_left5.npy', 'office/obj_left6.npy',
 					'office/obj_right4.npy', 'office/obj_right5.npy', 'office/obj_right6.npy']


train_images = []
train_labels = []
test_images = []
test_labels = []


for dirPath, dirNames, fileNames in os.walk(label_path):
	for i in range(len(train_label_file)):
		train_labels = np.hstack((train_labels, np.load(label_path + train_label_file[i])))

for dirPath, dirNames, fileNames in os.walk(label_path):
	for i in range(len(test_label_file)):
		test_labels = np.hstack((test_labels, np.load(label_path + test_label_file[i])))

for i in range(len(train_image_file)):
	for dirPath, dirNames, fileNames in os.walk(train_image_path + train_image_file[i]):
		for f in fileNames:
			train_images.append(os.path.join(dirPath, f))

for i in range(len(test_image_file)):
	for dirPath, dirNames, fileNames in os.walk(test_image_path + test_image_file[i]):
		for f in fileNames:
			test_images.append(os.path.join(dirPath, f))

train_images = sorted(train_images, key=lambda x: int(re.sub('\D', '', x)))
test_images = sorted(train_images, key=lambda x: int(re.sub('\D', '', x)))

NUM_BATCHES = (len(train_images) / BATCH_SIZE) + 1

# print(num_train_data)
# print(train_labels[0])

# data feeding preprocess
with tf.name_scope('ResNet'):
	x = tf.placeholder(dtype = tf.float32, shape = [None, INPUT_WIDTH * INPUT_HEIGHT * 3])
	x_image = tf.reshape(x, [-1, INPUT_HEIGHT, INPUT_WIDTH, 3])
	y_logits = resNet(x_image)
	print(y_logits)

# optimization
with tf.name_scope('Optimizer'):
	y_label = tf.placeholder("float", shape = [None, NUM_CLASS], name = 'y_label')
	loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_logits, labels= y_label))
	optimizer = tf.train.AdamOptimizer(LR).minimize(loss_func)

# caculate accuracy
with tf.name_scope('Accuracy_evaluation'):
	correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y_label, 1))
	accuracy = tf.cast(tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32)), dtype=tf.float32)


# start training
startTime = time()
init = tf.global_variables_initializer()

'''
# training
with tf.Session() as sess:	
	sess.run(init)
	saver = tf.train.Saver()
	for epoch in range(EPOCH):
		for i in range(NUM_BATCHES-1):
			batch_x = []
			batch_y = []
			for j in range(BATCH_SIZE):
				# print(epoch, i, j)
				image = Image.open(train_images[i * BATCH_SIZE + j])
				image_resized = cv.resize(np.asarray(image), (INPUT_HEIGHT, INPUT_WIDTH))
				image_resized = np.reshape(image_resized, (INPUT_HEIGHT * INPUT_WIDTH * 3))
				
				# one hot label
				index = int(train_labels[int(i* BATCH_SIZE + j)])
				oh_label = np.zeros(NUM_CLASS)
				oh_label[index] = 1
				
				batch_x.append(image_resized)
				batch_y.append(list(oh_label))
				# batch_y.append(train_labels[int(i* BATCH_SIZE + j)].astype(int))

			sess.run(optimizer, feed_dict = {x: batch_x, y_label: batch_y})
			if epoch % 5 == 0:
				save_path = saver.save(sess, './ckpt/obj_det_model.ckpt', global_step = epoch+1)

		val_x = []
		val_y = []
		# for k in range(len(test_images)): 
		for k in range(BATCH_SIZE):
			# print(epoch, k+epoch*30)
			t_image = Image.open(test_images[k + epoch * BATCH_SIZE])
			t_image_resized = cv.resize(np.asarray(t_image), (INPUT_HEIGHT, INPUT_WIDTH))
			t_image_resized = np.reshape(t_image_resized, (INPUT_HEIGHT * INPUT_WIDTH * 3))
			
			idx = int(test_labels[int(k + epoch * BATCH_SIZE)])
			te_label = np.zeros(NUM_CLASS)
			te_label[idx] = 1
			
			val_x.append(t_image_resized)
			val_y.append(list(te_label))
			# val_y.append(test_labels[int(i* BATCH_SIZE + j)].astype(int))
		
		loss, acc = sess.run([loss_func, accuracy], feed_dict={x: val_x, y_label: val_y})
		loss_list.append(loss)
		accuracy_list.append(acc)
		
		
		print("epoch({0:2d}):".format(epoch+1))
		print('loss     = {0:6.6f}'.format(loss))
		print('accuracy = {0:6.6f}'.format(acc))

# duration calculating
duration = time()-startTime
print('duration = ', duration)
'''

# testing and evaluation
test_x = []
test_y = []
y_score = []

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess, './obj_det_model.ckpt-46')

	test_acc = 0
	for j in range(0,len(test_images)):

		t_image = Image.open(test_images[j])
		t_image_resized = cv.resize(np.asarray(t_image), (INPUT_HEIGHT, INPUT_WIDTH))
		t_image_resized = np.reshape(t_image_resized, (-1, INPUT_HEIGHT * INPUT_WIDTH * 3))
		test_x.append(t_image_resized)
		y_score.append(sess.run(y_logits, feed_dict={x:t_image_resized}))

		idx = int(test_labels[j])
		te_label = np.zeros(NUM_CLASS)
		te_label[idx] = 1
		te_label = np.reshape(te_label, (-1, 24))
		test_y.append(list(te_label))
		
		test_acc = test_acc + sess.run(accuracy, feed_dict={x:t_image_resized, y_label:te_label})
		print(j)
	
	test_acc = test_acc / len(test_images)
	# print('accuracy = ', test_acc)

	'''
	# For each class
	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(NUM_CLASS):
		test_y = np.squeeze(np.asarray(test_y))
		y_score = np.squeeze(np.asarray(y_score))
		
		precision[i], recall[i], _ = precision_recall_curve(test_y[:, i], y_score[:, i])
		average_precision[i] = average_precision_score(test_y[:, i], y_score[:, i])

	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(test_y.ravel(),	y_score.ravel())
	average_precision["micro"] = average_precision_score(test_y, y_score, average="micro")
	print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))	

	plt.figure()
	plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
	plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('Average precision score, micro-averaged over all classes: AUC={0:0.2f}'.format(average_precision["micro"]))
	'''

	# setup plot details
	colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

	plt.figure(figsize=(7, 8))
	f_scores = np.linspace(0.2, 0.8, num=4)
	lines = []
	labels = []
	for f_score in f_scores:
		x = np.linspace(0.01, 1)
		y = f_score * x / (2 * x - f_score)
		l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
		plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

	lines.append(l)
	labels.append('iso-f1 curves')
	l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
	lines.append(l)
	labels.append('micro-average Precision-recall (area = {0:0.2f})'
		          ''.format(average_precision["micro"]))

	for i, color in zip(range(NUM_CLASS), colors):
		l, = plt.plot(recall[i], precision[i], color=color, lw=2)
		lines.append(l)
		labels.append('P-R for class {0} (area = {1:0.2f})'
		              ''.format(i, average_precision[i]))

	fig = plt.gcf()
	fig.subplots_adjust(bottom=0.25)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Extension of Precision-Recall curve to multi-class')
	plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=7))


	plt.show()


