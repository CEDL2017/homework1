import tensorflow as tf
import numpy as np

class AlexNet(object):
    """Create the graph of the AlexNet model.
    Adapted from: https://github.com/kratzert/finetune_alexnet_with_tensorflow
    """
    def __init__(self, x_1, x_2, keep_prob, num_classes, skip_layer):
        # Parse input arguments into class variables
        self.X = x_1
        self.X_HAND = x_1
        self.X_HEAD = x_2
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.WEIGHTS_PATH = 'pt_weight.npy'


        self.create_graph()
        # self.create_graph_two_stream()

    def create_graph(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        
        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        
        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    def create_graph_two_stream(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1_hand = conv(self.X_HAND, 11, 11, 96, 4, 4, padding='VALID', name='conv1_hand')
        norm1_hand = lrn(conv1_hand, 2, 2e-05, 0.75, name='norm1')
        pool1_hand = max_pool(norm1_hand, 3, 3, 2, 2, padding='VALID', name='pool1_hand')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2_hand = conv(pool1_hand, 5, 5, 256, 1, 1, groups=2, name='conv2_hand')
        norm2_hand = lrn(conv2_hand, 2, 2e-05, 0.75, name='norm2_hand')
        pool2_hand = max_pool(norm2_hand, 3, 3, 2, 2, padding='VALID', name='pool2_hand')

        # 3rd Layer: Conv (w ReLu)
        conv3_hand = conv(pool2_hand, 3, 3, 384, 1, 1, name='conv3_hand')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4_hand = conv(conv3_hand, 3, 3, 384, 1, 1, groups=2, name='conv4_hand')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5_hand = conv(conv4_hand, 3, 3, 256, 1, 1, groups=2, name='conv5_hand')
        pool5_hand = max_pool(conv5_hand, 3, 3, 2, 2, padding='VALID', name='pool5_hand')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened_hand = tf.reshape(pool5_hand, [-1, 6 * 6 * 256])
        fc6_hand = fc(flattened_hand, 6 * 6 * 256, 4096, name='fc6_hand')
        dropout6_hand = dropout(fc6_hand, self.KEEP_PROB)

        """Head"""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1_head = conv(self.X_HEAD, 11, 11, 96, 4, 4, padding='VALID', name='conv1_head')
        norm1_head = lrn(conv1_head, 2, 2e-05, 0.75, name='norm1_head')
        pool1_head = max_pool(norm1_head, 3, 3, 2, 2, padding='VALID', name='pool1_head')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2_head = conv(pool1_head, 5, 5, 256, 1, 1, groups=2, name='conv2_head')
        norm2_head = lrn(conv2_head, 2, 2e-05, 0.75, name='norm2_head')
        pool2_head = max_pool(norm2_head, 3, 3, 2, 2, padding='VALID', name='pool2_head')

        # 3rd Layer: Conv (w ReLu)
        conv3_head = conv(pool2_head, 3, 3, 384, 1, 1, name='conv3_head')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4_head = conv(conv3_head, 3, 3, 384, 1, 1, groups=2, name='conv4_head')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5_head = conv(conv4_head, 3, 3, 256, 1, 1, groups=2, name='conv5_head')
        pool5_head = max_pool(conv5_head, 3, 3, 2, 2, padding='VALID', name='pool5_head')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened_head = tf.reshape(pool5_head, [-1, 6 * 6 * 256])
        fc6_head = fc(flattened_head, 6 * 6 * 256, 4096, name='fc6_head')
        dropout6_head = dropout(fc6_head, self.KEEP_PROB)

        # 7th Layer: Concat -> FC (w ReLu) -> Dropout
        concat = tf.concat([dropout6_hand, dropout6_head], 1)
        fc7 = fc(concat, 8192, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    def load_initial_weights(self, session):
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

    def load_initial_weights_two_stream(self, session):
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:
                for str in ['_hand', '_head']:
                    with tf.variable_scope(op_name+str, reuse=True):

                        # Assign weights/biases to their corresponding tf variable
                        for data in weights_dict[op_name]:

                            # Biases
                            if len(data.shape) == 1:
                                var = tf.get_variable('biases', trainable=False)
                                session.run(var.assign(data))

                            # Weights
                            else:
                                var = tf.get_variable('weights', trainable=False)
                                session.run(var.assign(data))




def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    # Convolution Layer
    input_channels = int(x.get_shape()[-1])
    convolution = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolution(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolution them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolution(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    # Fully Connected Layer
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    # Max Pooling Layer
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    # Local Response Normalization
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    # Dropout
    return tf.nn.dropout(x, keep_prob)