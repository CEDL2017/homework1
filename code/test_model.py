import os
import numpy as np
from tensorflow.contrib.keras.python.keras.utils import np_utils
from create_models import *
from evaluate_acc import *


def predict_model_fc_features(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, fine_tune_model_weights_path, test_fc_features_path,
									img_height=None, img_width=None, dataset_loc='hand'):
	# load fine-tune model and output fc layer
	model = create_model(num_fc_neurons, dropout_rate, model_name, num_classes=num_classes, img_height=img_height, img_width=img_width, include_loc='all')
	model.load_weights(fine_tune_model_weights_path)
	
	# get fc features, concatenate base_model and top_model again
	while model.layers[-1].layers[-1].name != 'fc6':
		model.layers[-1].pop()
	top_model = Sequential(model.layers[-1].layers)
	model.layers.pop()
	model = Model(inputs=model.input, outputs=top_model(model.layers[-1].output))
	
	# extract fc features on test set and save
	test_fc_features = model.predict_generator(d_gen.test_data(dataset_loc=dataset_loc), d_gen.test_steps(), verbose=1)
	np.save(open(test_fc_features_path, 'wb'), test_fc_features)


def concatenate_hand_and_head_features(test_fc_hand_features_path, test_fc_head_features_path, concatenate_test_fc_features_path):
	test_fc_hand_features = np.load(open(test_fc_hand_features_path, 'rb'))
	test_fc_head_features = np.load(open(test_fc_head_features_path, 'rb'))
	np.save(open(concatenate_test_fc_features_path, 'wb'), np.append(test_fc_hand_features, test_fc_head_features, axis=1)) # concatenate fc features


def predict_two_stream_classifier(d_gen, num_fc_neurons, dropout_rate, num_classes, two_stream_classifier_weights_path, concatenate_test_fc_features_path, probas_pred_test_path):
	# concatenate two stream CNN
	classifier = create_two_stream_classifier(num_fc_neurons, dropout_rate, num_classes=num_classes)
	classifier.load_weights(two_stream_classifier_weights_path)
	
	# predict on test and save
	probas_pred_test = classifier.predict_generator(d_gen.test_data(dataset_loc='feature', features_path=concatenate_test_fc_features_path), d_gen.test_steps(), verbose=1)
	np.save(open(probas_pred_test_path, 'wb'), probas_pred_test)


def run_test_process(model_name, batch_size, epochs, num_fc_neurons, dropout_rate, zip_ref_frames, zip_ref_labels, output_path,
						mode='obj', setting_index=0, num_classes=24, img_height=None, img_width=None):
	if img_height == None or img_width == None: # default model input shape
		img_height, img_width = get_model_input_shape(model_name)
	
	# create DataGenerator
	d_gen = DataGenerator(zip_ref_frames, zip_ref_labels, img_height, img_width, batch_size=1)
	
	for dataset_loc in ['hand', 'head']:
		# settings for path
		fine_tune_model_weights_path = os.path.join(output_path, 'fine_tune_weights_'+model_name+'_'+dataset_loc+'_'+str(epochs)+'.h5')
		test_fc_features_path = os.path.join(output_path, 'test_fc_features_'+model_name+'_'+dataset_loc+'.npy')
		
		probas_pred_test = predict_model_fc_features(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, fine_tune_model_weights_path, test_fc_features_path,
									img_height, img_width, dataset_loc)
	
	# concatenate two stream output features
	test_fc_hand_features_path = os.path.join(output_path, 'test_fc_features_'+model_name+'_hand.npy')
	test_fc_head_features_path = os.path.join(output_path, 'test_fc_features_'+model_name+'_head.npy')
	concatenate_test_fc_features_path = os.path.join(output_path, 'concatenate_test_fc_features_'+model_name+'.npy')
	concatenate_hand_and_head_features(test_fc_hand_features_path, test_fc_head_features_path, concatenate_test_fc_features_path)
	
	two_stream_classifier_weights_path = os.path.join(output_path, 'two_stream_classifier_weights_'+model_name+'.h5')
	probas_pred_test_path = os.path.join(output_path, 'probas_pred_test_'+model_name+'.npy')
	predict_two_stream_classifier(d_gen, num_fc_neurons, dropout_rate, num_classes, two_stream_classifier_weights_path, concatenate_test_fc_features_path, probas_pred_test_path)
	
	# evaluate on test set
	probas_pred_test = np.load(open(probas_pred_test_path, 'rb')) # load test prediction (probability)
	pred_test_labels = np.argmax(probas_pred_test, axis=1).astype(int) # change probability to class label, axis=1 : row, determine predicted class number
	GT_test_labels = load_test_labels(zip_ref_labels, mode, setting_index=setting_index)
	GT_test_labels_one_hot = np_utils.to_categorical(GT_test_labels, num_classes=num_classes) # labels to one-hot vectors
	evaluate_acc(GT_test_labels, pred_test_labels)
	
	plot_precision_recall_curve(GT_test_labels_one_hot, probas_pred_test, num_classes)
	obj_classes = ['free', 'computer', 'cellphone', 'coin', 'ruler', 'thermos-bottle', 'whiteboard-pen', 'whiteboard-eraser', 'pen', 'cup',
				'remote-control-TV', 'remote-control-AC', 'switch', 'windows', 'fridge', 'cupboard', 'water-tap', 'toy', 'kettle', 'bottle',
				'cookie', 'book', 'magnet', 'lamp-switch'] # np.arange(num_classes)
	plot_confusion_matrix(GT_test_labels, pred_test_labels, classes=obj_classes, normalize=False, title='Confusion Matrix')
