import os
import numpy as np
import tensorflow as tf

def image_scaling(img):
    h, w, c = img.get_shape().as_list()
    scale = np.random.uniform(0.5, 1.2)
    nH, nW = [(int)(h * scale), (int)(w * scale)]

    img = tf.image.resize_images(img, [nH, nW])
    img.set_shape([nH, nW, c])

    return img

def image_mirroring(img):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)

    return img

def read_labeled_image_list(data_list):
    f = open(data_list, 'r')
    images = []
    heads = []
    labels = []

    for line in f:
        try:
            hand, head, FA, obj, ges = line[:-1].split(' ')
        except:
            print('error in read list')

        image = os.path.join(hand)
        head = os.path.join(head)

        if not tf.gfile.Exists(image):
            raise ValueError('Failed to find file: ' + image)
        if not tf.gfile.Exists(head):
            raise ValueError('Failed to find file: ' + head)

        images.append(image)
        heads.append(head)
        labels.append(int(obj))

    return images, heads, labels

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, img_mean):
    img_contents = tf.read_file(input_queue[0])
    head_contents = tf.read_file(input_queue[1])
    label = input_queue[2]

    img = tf.image.decode_png(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)

    head = tf.image.decode_png(head_contents, channels=3)
    head_r, head_g, head_b = tf.split(axis=2, num_or_size_splits=3, value=head)
    head = tf.cast(tf.concat(axis=2, values=[head_b, head_g, head_r]), dtype=tf.float32)

    # scale input image by factor 0.5
    h, w = input_size
    nH, nW = (500, 500)
    img = tf.image.resize_images(img, [nH, nW])
    img.set_shape([nH, nW, 3])
    head = tf.image.resize_images(head, [nH, nW])
    head.set_shape([nH, nW, 3])

    # Data augmentation
    if input_size is not None:
        if random_scale == True:
            img = image_scaling(img)
        if random_mirror == True:
            img = image_mirroring(img)

    return img, head, label

class ImageReader(object):
    def __init__(self, data_list, input_size, random_scale, random_mirror, img_mean, coord):
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.head_list, self.label_list = read_labeled_image_list(self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.heads = tf.convert_to_tensor(self.head_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.int32)
        self.queue = tf.train.slice_input_producer([self.images, self.heads, self.labels], shuffle=input_size is not None)
        self.image, self.head, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror, img_mean)

    def dequeue(self, num_elements):
        image_batch, head_batch, label_batch = tf.train.batch([self.image, self.head, self.label], num_elements)

        return image_batch, head_batch, label_batch
