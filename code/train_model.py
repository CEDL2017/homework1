from __future__ import print_function

import os
import numpy as np
from tensorflow.contrib.keras.python.keras import optimizers, callbacks
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, LambdaCallback
from create_models import *


def extract_pretrained_CNN_features(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, train_CNN_features_path,
									img_height=None, img_width=None, dataset_loc='hand'):
	# load pre-trained convolutional model
	model = create_model(num_fc_neurons, dropout_rate, model_name, num_classes=num_classes, img_height=img_height, img_width=img_width, include_loc='base')
	
	# extract CNN features and save
	train_features = model.predict_generator(d_gen.training_data(dataset_loc=dataset_loc), d_gen.training_steps()-1, verbose=1)
	validation_features = model.predict_generator(d_gen.validation_data(dataset_loc=dataset_loc), d_gen.validation_steps()-1, verbose=1)
	np.save(open(train_CNN_features_path, 'wb'), np.append(train_features, validation_features, axis=0)) # concatenate CNN features


def train_classifier(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, epochs, train_CNN_features_path, top_model_weights_path,
						img_height=None, img_width=None, dataset_loc='hand'):
	activation = 'softmax' #'sigmoid' if dataset_loc == 'head' else 'softmax'
	loss = 'categorical_crossentropy' #'binary_crossentropy' if dataset_loc == 'head' else 'categorical_crossentropy'
	
	# create classifier model
	model = create_model(num_fc_neurons, dropout_rate, model_name, num_classes=num_classes, img_height=img_height, img_width=img_width, include_loc='top', activation=activation)
	
	# train model and save weights
	model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-6), loss=loss, metrics=['accuracy']) # optimizer=optimizers.SGD(), Adam()
	model.fit_generator(d_gen.training_data(dataset_loc='feature', to_shuffle=True, features_path=train_CNN_features_path), d_gen.training_steps(), epochs=epochs,
							validation_data=d_gen.validation_data(dataset_loc='feature', features_path=train_CNN_features_path), validation_steps=d_gen.validation_steps())
	model.save_weights(top_model_weights_path)


def fine_tune(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, fixed_last_layer_name, epochs, top_model_weights_path, fine_tune_model_weights_path,
				pretrained_model_weights_path=None, img_height=None, img_width=None, dataset_loc='hand', callbacks_list=None):
	activation = 'softmax' #'sigmoid' if dataset_loc == 'head' else 'softmax'
	loss = 'categorical_crossentropy' #'binary_crossentropy' if dataset_loc == 'head' else 'categorical_crossentropy'
	
	# create complete model
	model = create_model(num_fc_neurons, dropout_rate, model_name, num_classes=num_classes, top_model_weights_path=top_model_weights_path, img_height=img_height, img_width=img_width, include_loc='all', activation=activation)
	if fixed_last_layer_name != '':
		fixed_num_layers = [l.name for l in model.layers].index(fixed_last_layer_name) + 1 # find index of model layer by name
	else:
		fixed_num_layers = 0
	
	# load pre-trained model weights
	if pretrained_model_weights_path != None:
		model.load_weights(pretrained_model_weights_path)
	
	# set the first several (convolutional) layers to non-trainable (weights will not be updated)
	for layer in model.layers[:fixed_num_layers]:
		layer.trainable = False
	
	# fine-tune model and save weights
	model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-6), loss=loss, metrics=['accuracy']) # optimizer=optimizers.SGD(), Adam()
	model.fit_generator(d_gen.training_data(dataset_loc=dataset_loc, to_shuffle=True), d_gen.training_steps(), epochs=epochs,
							validation_data=d_gen.validation_data(dataset_loc=dataset_loc), validation_steps=d_gen.validation_steps(), callbacks=callbacks_list) # verbose=0
	model.save_weights(fine_tune_model_weights_path)


def extract_model_fc_features(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, fine_tune_model_weights_path, train_fc_features_path,
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
	
	# extract fc features and save
	train_fc_features = model.predict_generator(d_gen.training_data(dataset_loc=dataset_loc), d_gen.training_steps()-1, verbose=1)
	validation_fc_features = model.predict_generator(d_gen.validation_data(dataset_loc=dataset_loc), d_gen.validation_steps()-1, verbose=1)
	np.save(open(train_fc_features_path, 'wb'), np.append(train_fc_features, validation_fc_features, axis=0)) # concatenate fc features



def concatenate_hand_and_head_features(train_fc_hand_features_path, train_fc_head_features_path, concatenate_train_fc_features_path):
	train_fc_hand_features = np.load(open(train_fc_hand_features_path, 'rb'))
	train_fc_head_features = np.load(open(train_fc_head_features_path, 'rb'))
	np.save(open(concatenate_train_fc_features_path, 'wb'), np.append(train_fc_hand_features, train_fc_head_features, axis=1)) # concatenate fc features


def train_two_stream_classifier(d_gen, num_fc_neurons, dropout_rate, num_classes, epochs, concatenate_train_fc_features_path, two_stream_classifier_weights_path,
									callbacks_list=None):
	# concatenate two stream CNN
	classifier = create_two_stream_classifier(num_fc_neurons, dropout_rate, num_classes=num_classes)
	
	# train classifier and save weights
	classifier.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy']) # optimizer=optimizers.SGD(), Adam()
	classifier.fit_generator(d_gen.training_data(dataset_loc='feature', to_shuffle=True, features_path=concatenate_train_fc_features_path), d_gen.training_steps(), epochs=epochs,
								validation_data=d_gen.validation_data(dataset_loc='feature', features_path=concatenate_train_fc_features_path), validation_steps=d_gen.validation_steps(), callbacks=callbacks_list) # verbose=0
	classifier.save_weights(two_stream_classifier_weights_path)





def run_train_process(model_name, batch_size, epochs, validation_split, num_fc_neurons, dropout_rate, zip_ref_frames, zip_ref_labels, output_path,
						mode='obj', setting_index=0, num_classes=24, img_height=None, img_width=None, pretrained_model_weights_path=None):
	if img_height == None or img_width == None: # default model input shape
		img_height, img_width = get_model_input_shape(model_name)
	
	# create DataGenerator
	d_gen = DataGenerator(zip_ref_frames, zip_ref_labels, img_height, img_width, batch_size=batch_size, validation_split=validation_split)
	
	for dataset_loc in ['hand', 'head']:
		# settings for path
		train_CNN_features_path = os.path.join(output_path, 'train_CNN_features_'+model_name+'_'+dataset_loc+'.npy')
		top_model_weights_path = os.path.join(output_path, 'top_model_weights_'+model_name+'_'+dataset_loc+'.h5')
		fine_tune_model_weights_path = os.path.join(output_path, 'fine_tune_weights_'+model_name+'_'+dataset_loc+'_'+str(epochs)+'.h5')
		train_fc_features_path = os.path.join(output_path, 'train_fc_features_'+model_name+'_'+dataset_loc+'.npy')
		
		# build callback list, monitor on val_acc
		filepath = os.path.join(output_path, model_name+'_'+dataset_loc+'-{epoch:02d}.h5')
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
		batch_print_callback = LambdaCallback(on_epoch_end=lambda batch, logs: print('\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %(batch, logs['acc'], batch, logs['val_acc'])))
		#early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1)
		callbacks_list = [checkpoint, batch_print_callback]
		"""
		if model_name in ['ResNet50', 'VGG16']: # pre-trained model
			#extract_pretrained_CNN_features(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, train_CNN_features_path,
			#								img_height, img_width, dataset_loc)
			train_classifier(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, epochs, train_CNN_features_path, top_model_weights_path,
								img_height, img_width, dataset_loc)
		"""
		fixed_last_layer_name = {'AlexNet': '', 'ResNet50': 'res2c_branch2c', 'VGG16': 'block2_pool'} # for fine-tune process
		fine_tune(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, fixed_last_layer_name[model_name], epochs, None, fine_tune_model_weights_path,
					pretrained_model_weights_path, img_height, img_width, dataset_loc, callbacks_list)
		
		extract_model_fc_features(model_name, d_gen, num_fc_neurons, dropout_rate, num_classes, fine_tune_model_weights_path, train_fc_features_path,
										img_height, img_width, dataset_loc)
	
	
	# concatenate two stream output features
	train_fc_hand_features_path = os.path.join(output_path, 'train_fc_features_'+model_name+'_hand.npy')
	train_fc_head_features_path = os.path.join(output_path, 'train_fc_features_'+model_name+'_head.npy')
	concatenate_train_fc_features_path = os.path.join(output_path, 'concatenate_train_fc_features_'+model_name+'.npy')
	concatenate_hand_and_head_features(train_fc_hand_features_path, train_fc_head_features_path, concatenate_train_fc_features_path)
	
	# build callback list, monitor on val_acc
	filepath = os.path.join(output_path, model_name+'_two_stream-{epoch:02d}.h5')
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
	batch_print_callback = LambdaCallback(on_epoch_end=lambda batch, logs: print('\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %(batch, logs['acc'], batch, logs['val_acc'])))
	#early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1)
	callbacks_list = [checkpoint, batch_print_callback]
	
	two_stream_classifier_weights_path = os.path.join(output_path, 'two_stream_classifier_weights_'+model_name+'.h5')
	train_two_stream_classifier(d_gen, num_fc_neurons, dropout_rate, num_classes, epochs, concatenate_train_fc_features_path, two_stream_classifier_weights_path,
									callbacks_list)
