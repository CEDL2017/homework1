import os
import numpy as np
from PIL import Image
import sys
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from itertools import cycle


# for local
train_image_path = "./handcam/frames/train/"
test_image_path = "./handcam/frames/test/"
label_path = "./handcam/labels/"


test_image_file = ['house/1/Lhand', 'house/2/Lhand', 'house/3/Lhand', 
					'house/1/Rhand', 'house/2/Rhand', 'house/3/Rhand',
					'lab/1/Lhand', 'lab/2/Lhand', 'lab/3/Lhand', 'lab/4/Lhand',
					'lab/1/Rhand', 'lab/2/Rhand', 'lab/3/Rhand', 'lab/4/Rhand',
					'office/1/Lhand', 'office/2/Lhand', 'office/3/Lhand', 
					'office/1/Rhand', 'office/2/Rhand', 'office/3/Rhand',]


test_label_file = ['house/obj_left4.npy', 'house/obj_left5.npy', 'house/obj_left6.npy',
					'house/obj_right4.npy', 'house/obj_right5.npy', 'house/obj_right6.npy',
	 				'lab/obj_left5.npy', 'lab/obj_left6.npy', 'lab/obj_left7.npy', 'lab/obj_left8.npy',
	 				'lab/obj_right5.npy', 'lab/obj_right6.npy', 'lab/obj_right7.npy', 'lab/obj_right8.npy',
 					'office/obj_left4.npy', 'office/obj_left5.npy', 'office/obj_left6.npy',
 					'office/obj_right4.npy', 'office/obj_right5.npy', 'office/obj_right6.npy']

test_images = []
test_labels = []

for dirPath, dirNames, fileNames in os.walk(label_path):
	for i in range(len(test_label_file)):
		test_labels = np.hstack((test_labels, np.load(label_path + test_label_file[i])))

for i in range(len(test_image_file)):
	for dirPath, dirNames, fileNames in os.walk(test_image_path + test_image_file[i]):
		for f in fileNames:
			test_images.append(os.path.join(dirPath, f))

test_images = sorted(train_images, key=lambda x: int(re.sub('\D', '', x)))

test_x = []
test_y = []
y_score = []

with tf.Session() as sess:

	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
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
		# print(j)
	
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
