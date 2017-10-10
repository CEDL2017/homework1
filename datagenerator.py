import tensorflow as tf
import numpy as np

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000):
        """Create a new ImageDataGenerator.
        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.
        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """
        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths_hand = convert_to_tensor(self.img_paths_hand, dtype=dtypes.string)
        self.img_paths_head = convert_to_tensor(self.img_paths_head, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths_hand, self.img_paths_head, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_threads=8,
                      output_buffer_size=100*batch_size)

        elif mode == 'validation':
            data = data.map(self._parse_function_validation, num_threads=8,
                      output_buffer_size=100*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths_hand = []
        self.img_paths_head = []
        self.labels = []
        
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                str1, str2, str3, str4, str5 = line[:-1].split(' ')
                self.img_paths_hand.append(str1)
                self.img_paths_head.append(str2)
                self.labels.append(int(str4))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path_hand = self.img_paths_hand
        path_head = self.img_paths_head
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths_hand = []
        self.img_paths_head = []
        self.labels = []
        for i in permutation:
            self.img_paths_hand.append(path_hand[i])
            self.img_paths_head.append(path_head[i])
            self.labels.append(labels[i])
            
    def _pre_process(self, images):
        if tf.image.random_flip_left_right:
            images = tf.image.random_flip_left_right(images)
        if tf.image.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if tf.image.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        new_size = tf.constant([224,224], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        return images

    def _parse_function_train(self, filename_hand, filename_head, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the hand image
        img_string_hand = tf.read_file(filename_hand)
        img_decoded_hand = tf.image.decode_png(img_string_hand, channels=3)
        img_resized_hand = tf.image.resize_images(img_decoded_hand, [224, 224])
        
        """
        Dataaugmentation comes here.
        """
        img_aug_hand =  self._pre_process(img_resized_hand)
        
        img_centered_hand = tf.subtract(img_aug_hand, VGG_MEAN)

        # RGB -> BGR
        img_bgr_hand = img_centered_hand[:, :, ::-1]

        # load and preprocess the head image
        img_string_head = tf.read_file(filename_head)
        img_decoded_head = tf.image.decode_png(img_string_head, channels=3)
        img_resized_head = tf.image.resize_images(img_decoded_head, [224, 224])
        
        """
        Dataaugmentation comes here.
        """
        img_aug_head =  self._pre_process(img_resized_head)
        
        img_centered_head = tf.subtract(img_aug_head, VGG_MEAN)

        # RGB -> BGR
        img_bgr_head = img_centered_head[:, :, ::-1]

        return img_bgr_hand, img_bgr_head, one_hot

    def _parse_function_validation(self, filename_hand, filename_head, label):
        """Input parser for samples of the validation set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the hand image
        img_string_hand = tf.read_file(filename_hand)
        img_decoded_hand = tf.image.decode_png(img_string_hand, channels=3)
        img_resized_hand = tf.image.resize_images(img_decoded_hand, [224, 224])
        img_centered_hand = tf.subtract(img_resized_hand, VGG_MEAN)

        # RGB -> BGR
        img_bgr_hand = img_centered_hand[:, :, ::-1]

        # load and preprocess the head image
        img_string_head = tf.read_file(filename_head)
        img_decoded_head = tf.image.decode_png(img_string_head, channels=3)
        img_resized_head = tf.image.resize_images(img_decoded_head, [224, 224])
        img_centered_head = tf.subtract(img_resized_head, VGG_MEAN)

        # RGB -> BGR
        img_bgr_head = img_centered_head[:, :, ::-1]

        return img_bgr_hand, img_bgr_head, one_hot