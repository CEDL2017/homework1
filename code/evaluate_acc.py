import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, average_precision_score, confusion_matrix


def evaluate_acc(GT, pred):
	GT_seq = np.zeros((0))
	pred_seq = np.zeros((0))
	
	for i in range(len(pred)):
		GT_seq = np.append(GT_seq, GT[i])
		pred_seq = np.append(pred_seq, pred[i])
	
	correct = np.count_nonzero(GT_seq == pred_seq)
	total = len(GT_seq)
	acc = (correct/float(total))*100
	print("\ntesting acc. = %.2f%%, %d/%d" %(acc, correct, total))


def plot_precision_recall_curve(GT_one_hot, probas_pred, num_classes):
	# For each class
	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(num_classes):
		precision[i], recall[i], _ = precision_recall_curve(GT_one_hot[:, i], probas_pred[:, i])
		average_precision[i] = average_precision_score(GT_one_hot[:, i], probas_pred[:, i])
	
	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(GT_one_hot.ravel(), probas_pred.ravel())
	average_precision["micro"] = average_precision_score(GT_one_hot, probas_pred, average="micro")
	
	# Plot Precision-Recall curve for each class
	lines = []
	labels = []
	plt.figure(figsize=(16,9))
	l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
	lines.append(l)
	labels.append('micro-avg (area = {0:0.2f})'
					''.format(average_precision["micro"]))
	
	for i in range(num_classes):
		l, = plt.plot(recall[i], precision[i], lw=2)
		lines.append(l)
		labels.append('class {0} (area = {1:0.2f})'
						''.format(i, average_precision[i]))
	
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Extension of Precision-Recall curve to multi-class')
	plt.legend(lines, labels, loc=(1.01, 0), prop=dict(size=8))
	plt.show()


def plot_confusion_matrix(GT, pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting 'normalize=True'.
	"""
	cm = confusion_matrix(GT, pred)
	
	plt.figure(figsize=(16,9))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	if not normalize:
		plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)
	
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	
	#print(cm)
	
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if cm[i, j] > thresh:
			color = "white"
		elif cm[i, j] > (GT.shape[0] / len(classes)) and i != j:
			color = "red"
		else:
			color = "black"
		plt.text(j, i, np.around(cm[i, j], decimals=2), horizontalalignment="center", color=color)
	
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
