import os
import math
import numpy as np
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.models import Sequential, Model
from tensorflow.contrib.keras.python.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.contrib.keras.python.keras.layers.core import Lambda
from tensorflow.contrib.keras.python.keras.utils import np_utils
from tensorflow.contrib.keras.python.keras.applications import ResNet50, VGG16
from load_datas import *


def get_model_input_shape(model_name):
	if model_name in ['AlexNet', 'ResNet50', 'VGG16']:
		img_height, img_width = 224, 224
	else:
		raise ValueError('Only AlexNet, ResNet50 and VGG16 can be created.')
	return img_height, img_width


def get_input_shape(img_height, img_width):
	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_height, img_width)
	else:
		input_shape = (img_height, img_width, 3)
	return input_shape


def LRN2D(k=2, n=5, alpha=1e-4, beta=0.75, **kwargs): # Local Response Normalization
	def f_lrn(X):
		samples, rows, cols, channels = X.get_shape()
		half_n = n // 2
		X_square = K.square(X)
		
		# pad with zeros (half_n channels)
		X_square_padded = K.spatial_2d_padding(K.permute_dimensions(X_square, (1, 0, 3, 2)), ((0, 0), (half_n, half_n))) # pad 2nd and 3rd dim
		X_square_padded = K.permute_dimensions(X_square_padded, (1, 0, 3, 2))
		
		# sum runs over n "adjacent" kernel maps at the same spatial position
		sum = 0
		for i in range(n):
			sum += X_square_padded[:, :, :, i:(i+int(channels))]
		scale = (k + alpha * sum) ** beta
		return X / scale
	
	return Lambda(f_lrn, **kwargs)


def create_AlexNet(num_fc_neurons, dropout_rate, num_classes=24, img_height=224, img_width=224, include_loc='all', activation='softmax'):
	weight_decay = 0.0005
	kernel_regularizer = regularizers.l2(weight_decay)
	bias_regularizer = regularizers.l2(weight_decay)
	
	# build a convolutional model
	base_model = Sequential()
	base_model.add(Conv2D(96, (11,11), strides=(4,4), padding='valid', activation='relu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name='conv1', input_shape=get_input_shape(img_height,img_width)))
	base_model.add(LRN2D(name='lrn1'))
	base_model.add(MaxPooling2D((3,3), strides=(2,2), name='pool1'))
	
	base_model.add(Conv2D(256, (5,5), strides=(1,1), padding='same', activation='relu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name='conv2'))
	base_model.add(LRN2D(name='lrn2'))
	base_model.add(MaxPooling2D((3,3), strides=(2,2), name='pool2'))
	
	base_model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name='conv3'))
	base_model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name='conv4'))
	base_model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name='conv5'))
	base_model.add(MaxPooling2D((3,3), strides=(2,2), name='pool3'))
	
	# build a classifier model to put on top of the convolutional model
	top_model = Sequential()
	top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
	for i in range(6,8):
		top_model.add(Dense(num_fc_neurons, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name='fc'+str(i)))
		#top_model.add(BatchNormalization(axis=1, name='fc'+str(i)+'_bn'))
		top_model.add(Activation('relu', name='fc'+str(i)+'_ac'))
		top_model.add(Dropout(dropout_rate))
	top_model.add(Dense(num_classes, activation=activation, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, name='predictions'))
	
	if include_loc == 'base':
		model = base_model
	elif include_loc == 'top':
		model = top_model
	elif include_loc == 'all': # add the model on top of the convolutional base
		model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
	else:
		raise ValueError('Only "base", "top" and "all" can be included.')
	return model


def create_ResNet50(num_fc_neurons, dropout_rate, num_classes=24, top_model_weights_path=None, img_height=224, img_width=224, include_loc='all', activation='softmax'):
	# load pre-trained convolutional model
	base_model = ResNet50(weights='imagenet', include_top=False, input_shape=get_input_shape(img_height,img_width))
	
	# build a classifier model to put on top of the convolutional model
	top_model = Sequential()
	top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
	for i in range(6,8):
		top_model.add(Dense(num_fc_neurons, name='fc'+str(i)))
		#top_model.add(BatchNormalization(axis=1, name='fc'+str(i)+'_bn'))
		top_model.add(Activation('relu', name='fc'+str(i)+'_ac'))
		top_model.add(Dropout(dropout_rate))
	top_model.add(Dense(num_classes, activation=activation, name='predictions'))
	if top_model_weights_path != None:
		top_model.load_weights(top_model_weights_path)
	
	if include_loc == 'base':
		model = base_model
	elif include_loc == 'top':
		model = top_model
	elif include_loc == 'all': # add the model on top of the convolutional base
		model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
	else:
		raise ValueError('Only "base", "top" and "all" can be included.')
	return model


def create_VGG16(num_fc_neurons, dropout_rate, num_classes=24, top_model_weights_path=None, img_height=224, img_width=224, include_loc='all', activation='softmax'):
	# load pre-trained convolutional model
	base_model = VGG16(weights='imagenet', include_top=False, input_shape=get_input_shape(img_height,img_width))
	
	# build a classifier model to put on top of the convolutional model
	top_model = Sequential()
	top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
	for i in range(6,8):
		top_model.add(Dense(num_fc_neurons, name='fc'+str(i)))
		#top_model.add(BatchNormalization(axis=1, name='fc'+str(i)+'_bn'))
		top_model.add(Activation('relu', name='fc'+str(i)+'_ac'))
		top_model.add(Dropout(dropout_rate))
	top_model.add(Dense(num_classes, activation=activation, name='predictions'))
	if top_model_weights_path != None:
		top_model.load_weights(top_model_weights_path)
	
	if include_loc == 'base':
		model = base_model
	elif include_loc == 'top':
		model = top_model
	elif include_loc == 'all': # add the model on top of the convolutional base
		model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
	else:
		raise ValueError('Only "base", "top" and "all" can be included.')
	return model



"""
create_model parameters:
model_name = ['AlexNet', 'ResNet50'(pre-trained), 'VGG16'(pre-trained)]
include_loc = ['base'(convolution), 'top'(classifier), 'all']
"""
def create_model(num_fc_neurons, dropout_rate, model_name, num_classes=24, top_model_weights_path=None, img_height=None, img_width=None, include_loc='all', activation='softmax'):
	if img_height == None or img_width == None: # default model input shape
		img_height, img_width = get_model_input_shape(model_name)
	
	if model_name == 'AlexNet':
		model = create_AlexNet(num_fc_neurons, dropout_rate, num_classes=num_classes, img_height=img_height, img_width=img_width, include_loc=include_loc, activation=activation)
	elif model_name == 'ResNet50':
		model = create_ResNet50(num_fc_neurons, dropout_rate, num_classes=num_classes, top_model_weights_path=top_model_weights_path, img_height=img_height, img_width=img_width, include_loc=include_loc, activation=activation)
	elif model_name == 'VGG16':
		model = create_VGG16(num_fc_neurons, dropout_rate, num_classes=num_classes, top_model_weights_path=top_model_weights_path, img_height=img_height, img_width=img_width, include_loc=include_loc, activation=activation)
	else:
		raise ValueError('Only AlexNet, ResNet50 and VGG16 can be created.')
	#print(model.summary())
	return model





def create_two_stream_classifier(num_fc_neurons, dropout_rate, num_classes=24): # classifier_weights_path=None
	classifier = Sequential()
	classifier.add(Dense(num_fc_neurons, name='fc7', input_shape=(num_fc_neurons*2,)))
	#classifier.add(BatchNormalization(axis=1, name='fc7_bn'))
	classifier.add(Activation('relu', name='fc7_ac'))
	classifier.add(Dropout(dropout_rate))
	classifier.add(Dense(num_classes, activation='softmax', name='predictions'))
	return classifier







class DataGenerator:
	def __init__(self, zip_ref_frames, zip_ref_labels, img_height, img_width, batch_size=32, validation_split=0.2):
		self.zip_ref_frames = zip_ref_frames
		self.zip_ref_labels = zip_ref_labels
		self.img_height = img_height
		self.img_width = img_width
		self.batch_size = batch_size
		self.validation_split = validation_split
		self.train_hand_name_array, self.train_head_name_array = get_image_name_array(self.zip_ref_frames, 'train')
		self.test_hand_name_array, self.test_head_name_array = get_image_name_array(self.zip_ref_frames, 'test')
		self.num_validation_data = int(math.floor(self.train_hand_name_array.size * validation_split))
		self.num_training_data = self.train_hand_name_array.size - self.num_validation_data
		self.num_test_data = self.test_hand_name_array.size
		self.num_training_features = (self.training_steps()-1) * self.batch_size
		self.num_validation_features = (self.validation_steps()-1) * self.batch_size
		self.num_test_features = (self.test_steps()-1) * self.batch_size
	
	def training_data(self, dataset_loc='hand', to_shuffle=False, features_path=None):
		return self.data_generator(dataset_name='train', batch_size=self.batch_size, dataset_loc=dataset_loc, to_shuffle=to_shuffle, features_path=features_path)
	
	def training_steps(self):
		return int(math.ceil(float(self.num_training_data) / self.batch_size))
	
	def validation_data(self, dataset_loc='hand', to_shuffle=False, features_path=None):
		return self.data_generator(dataset_name='validation', batch_size=self.batch_size, dataset_loc=dataset_loc, to_shuffle=to_shuffle, features_path=features_path)
	
	def validation_steps(self):
		return int(math.ceil(float(self.num_validation_data) / self.batch_size))
	
	def test_data(self, dataset_loc='hand', to_shuffle=False, features_path=None):
		return self.data_generator(dataset_name='test', batch_size=self.batch_size, dataset_loc=dataset_loc, to_shuffle=to_shuffle, features_path=features_path)
	
	def test_steps(self):
		return int(math.ceil(float(self.num_test_data) / self.batch_size))
	
	def data_generator(self, mode='obj', setting_index=0, num_classes=24, dataset_name='train', batch_size=32, dataset_loc='hand', to_shuffle=False, features_path=None):
		for_head = False #True if dataset_loc == 'head' else False
		
		if dataset_name in ['train', 'validation']:
			GT_train_labels = load_train_labels(self.zip_ref_labels, mode, setting_index=setting_index, for_head=for_head)
			if dataset_loc == 'hand':
				image_name_array = self.train_hand_name_array
			elif dataset_loc == 'head':
				image_name_array = self.train_head_name_array
				#GT_train_labels = GT_train_labels[0,:] + GT_train_labels[1,:]*num_classes
			else: # load features
				train_features = np.load(open(features_path, 'rb'))
				GT_train_labels = np.append(GT_train_labels[:self.num_training_features], GT_train_labels[self.num_training_data:(self.num_training_data+self.num_validation_features)])
		else: # test set
			GT_test_labels = load_test_labels(self.zip_ref_labels, mode, setting_index=setting_index, for_head=for_head)
			if dataset_loc == 'hand':
				image_name_array = self.test_hand_name_array
			elif dataset_loc == 'head':
				image_name_array = self.test_head_name_array
				#GT_test_labels = GT_test_labels[0,:] + GT_test_labels[1,:]*num_classes
			else: # load features
				test_features = np.load(open(features_path, 'rb'))
				GT_test_labels = GT_test_labels[:self.num_test_features]
		
		if dataset_name == 'train':
			if dataset_loc in ['hand', 'head']:
				dataset_size = self.num_training_data
				x = image_name_array[:self.num_training_data]
				y = GT_train_labels[:self.num_training_data]
			else: # load features
				dataset_size = self.num_training_features
				x = train_features[:self.num_training_features]
				y = GT_train_labels[:self.num_training_features]
		elif dataset_name == 'validation':
			if dataset_loc in ['hand', 'head']:
				dataset_size = self.num_validation_data
				x = image_name_array[self.num_training_data:]
				y = GT_train_labels[self.num_training_data:]
			else: # load features
				dataset_size = self.num_validation_features
				x = train_features[self.num_training_features:]
				y = GT_train_labels[self.num_training_features:]
		else: # test set
			if dataset_loc in ['hand', 'head']:
				dataset_size = self.num_test_data
				x = image_name_array
				y = GT_test_labels
			else: # load features
				dataset_size = self.num_test_features
				x = test_features
				y = GT_test_labels
		
		if to_shuffle: # shuffle in mini-batch
			if dataset_name in ['validation', 'test']:
				prng = np.random.RandomState(1234) # random seed
			else: # train set
				prng = np.random
			rand_perm_array = prng.permutation(dataset_size)
			x = x[rand_perm_array]
			y = y[rand_perm_array]
		
		if for_head: # multi-label
			y_to_categorical = np.zeros((y.size, num_classes))
			y_to_categorical[range(y.size), (y % num_classes)] = 1
			y_to_categorical[range(y.size), (y // num_classes)] = 1
			y = y_to_categorical
		else:
			y = np_utils.to_categorical(y, num_classes=num_classes) # labels to one-hot vectors
		
		i = 0
		while True:
			if i + batch_size > dataset_size:
				i = 0
			
			if dataset_loc in ['hand', 'head']:
				x_batch = load_images(self.zip_ref_frames, self.img_height, self.img_width, image_name_array=x[i:(i+batch_size)])
			else: # load features
				x_batch = x[i:(i+batch_size)]
			y_batch = y[i:(i+batch_size)]
			yield x_batch, y_batch
			
			i += batch_size
