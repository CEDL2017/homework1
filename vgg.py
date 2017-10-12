import tensorflow as tf
import numpy as np


class VGGNet(object):
    """Create the graph of the VGGNet model.
    """

    def __init__(self, x_1, x_2, keep_prob, num_classes, skip_layer):
        # Parse input arguments into class variables
        self.X = x_1
        self.X_HAND = x_1
        self.X_HEAD = x_2
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.WEIGHTS_PATH = 'vgg_weight.npy'

        # self.create_graph()
        self.create_graph_two_stream()

    def create_graph(self, is_pretrain=True):
        """Create the network graph."""
        x = conv('conv1_1', self.X, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = conv('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = conv('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = conv('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = conv('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = conv('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = conv('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = conv('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = conv('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = conv('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = conv('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = conv('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = conv('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = FC_layer('fc6', x, out_nodes=4096)
        x = tf.nn.relu(x)
        x = FC_layer('fc7', x, out_nodes=4096)
        x = tf.nn.relu(x)
        x = FC_layer('fc8', x, out_nodes=self.NUM_CLASSES)

        self.fc8 = x

    def create_graph_two_stream(self, is_pretrain=True):
        """Create the network graph."""
        x_hand = conv('conv1_1_hand', self.X_HAND, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = conv('conv1_2_hand', x_hand, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = pool('pool1_hand', x_hand, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x_hand = conv('conv2_1_hand', x_hand, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = conv('conv2_2_hand', x_hand, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = pool('pool2_hand', x_hand, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x_hand = conv('conv3_1_hand', x_hand, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = conv('conv3_2_hand', x_hand, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = conv('conv3_3_hand', x_hand, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = pool('pool3_hand', x_hand, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x_hand = conv('conv4_1_hand', x_hand, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = conv('conv4_2_hand', x_hand, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = conv('conv4_3_hand', x_hand, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = pool('pool3_hand', x_hand, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x_hand = conv('conv5_1_hand', x_hand, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = conv('conv5_2_hand', x_hand, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = conv('conv5_3_hand', x_hand, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_hand = pool('pool3_hand', x_hand, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        # flattened_hand = reshape(x_hand, [-1])

        """Head"""
        x_head = conv('conv1_1_head', self.X_HEAD, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = conv('conv1_2_head', x_head, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = pool('pool1_head', x_head, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x_head = conv('conv2_1_head', x_head, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = conv('conv2_2_head', x_head, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = pool('pool2_head', x_head, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x_head = conv('conv3_1_head', x_head, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = conv('conv3_2_head', x_head, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = conv('conv3_3_head', x_head, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = pool('pool3_head', x_head, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x_head = conv('conv4_1_head', x_head, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = conv('conv4_2_head', x_head, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = conv('conv4_3_head', x_head, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = pool('pool3_head', x_head, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x_head = conv('conv5_1_head', x_head, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = conv('conv5_2_head', x_head, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = conv('conv5_3_head', x_head, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x_head = pool('pool3_head', x_head, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        # flattened_head = reshape(x_head, [-1])

        """Merge"""
        concat = tf.concat([x_hand, x_head], 3)
        x = FC_layer('fc6', concat, out_nodes=4096)
        x = tf.nn.dropout(x, self.KEEP_PROB)
        x = FC_layer('fc7', x, out_nodes=4096)
        x = tf.nn.dropout(x, self.KEEP_PROB)
        x = FC_layer('fc8', x, out_nodes=self.NUM_CLASSES, relu=False)

        self.fc8 = x

    def load_initial_weights(self, session):
        data_dict = np.load(self.WEIGHTS_PATH, encoding='latin1').item()
        for key in data_dict:
            if key not in self.SKIP_LAYER:
                with tf.variable_scope(key, reuse=True):
                    for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                        session.run(tf.get_variable(subkey).assign(data))

    def load_initial_weights_two_stream(self, session):
        data_dict = np.load(self.WEIGHTS_PATH, encoding='latin1').item()
        for key in data_dict:
            if key not in self.SKIP_LAYER:
                for str in ['_hand', '_head']:
                    with tf.variable_scope(key+str, reuse=True):
                        for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                            session.run(tf.get_variable(subkey).assign(data))


def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()) # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x

def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x

def batch_norm(x):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


def FC_layer(layer_name, x, out_nodes, relu=True):
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        if relu:
            x = tf.nn.relu(x)
        return x

