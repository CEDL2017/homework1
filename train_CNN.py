import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import re
import random
import time

data_env= '..'
frames_path=os.path.join(data_env,'frames');
label_path=os.path.join(data_env,'labels');

def load_data(start_path):
    X = []
    FA = []
    ges = []
    obj = []
    head = []
    for dir_name in os.listdir(start_path): # 'office', 'lab', 'house'
        loc_path = os.path.join(start_path, dir_name)
        for num in os.listdir(loc_path): # '1', '2', '3'
            loc_num_path = os.path.join(loc_path, num)
            for pos in os.listdir(loc_num_path): # 'Lhand', 'Rhand', 'head'
                loc_num_pos_path = os.path.join(loc_num_path, pos)
                file_names = os.listdir(loc_num_pos_path)
                file_names = sorted(file_names, key=lambda x: int(re.sub('\D', '', x)))
                file_names = [os.path.join(loc_num_pos_path,f) for f in file_names]
                if pos[0] == 'L':
                    X += file_names
                    if 'train' in start_path:
                        FA += np.load(os.path.join(label_path, dir_name, 'FA_left'+num+'.npy')).astype(int).tolist()
                        ges += np.load(os.path.join(label_path, dir_name, 'ges_left'+num+'.npy')).astype(int).tolist()
                        obj += np.load(os.path.join(label_path, dir_name, 'obj_left'+num+'.npy')).astype(int).tolist()
                    elif 'test' in start_path:
                        if 'lab' in loc_num_pos_path:
                            offset = 4
                        else:
                            offset = 3
                        FA += np.load(os.path.join(label_path, dir_name, 'FA_left'+str(int(num)+offset)+'.npy')).astype(int).tolist()
                        ges += np.load(os.path.join(label_path, dir_name, 'ges_left'+str(int(num)+offset)+'.npy')).astype(int).tolist()
                        obj += np.load(os.path.join(label_path, dir_name, 'obj_left'+str(int(num)+offset)+'.npy')).astype(int).tolist()
                elif pos[0] == 'R':
                    X += file_names
                    if 'train' in start_path:
                        FA += np.load(os.path.join(label_path, dir_name, 'FA_right'+num+'.npy')).astype(int).tolist()
                        ges += np.load(os.path.join(label_path, dir_name, 'ges_right'+num+'.npy')).astype(int).tolist()
                        obj += np.load(os.path.join(label_path, dir_name, 'obj_right'+num+'.npy')).astype(int).tolist()
                    elif 'test' in start_path:
                        if 'lab' in loc_num_pos_path:
                            offset = 4
                        else:
                            offset = 3
                        FA += np.load(os.path.join(label_path, dir_name, 'FA_right'+str(int(num)+offset)+'.npy')).astype(int).tolist()
                        ges += np.load(os.path.join(label_path, dir_name, 'ges_right'+str(int(num)+offset)+'.npy')).astype(int).tolist()
                        obj += np.load(os.path.join(label_path, dir_name, 'obj_right'+str(int(num)+offset)+'.npy')).astype(int).tolist()
                else:
                    head += file_names
                    head += file_names
    
    assert len(X) == len(FA) == len(ges) == len(obj) == len(head)
    return np.array(X), np.array(FA), np.array(ges), np.array(obj), np.array(head)

class Model(object):
    def __init__(self, vgg19_npy_path, name='cedlCNN'):
        self.name = name
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        with tf.variable_scope(name):
            self.build_model()
    
    def build_model(self):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        
        start_time = time.time()
        print("build model started")
        # label
        self.FA = tf.placeholder(dtype=tf.int32, shape=[None])
        self.ges = tf.placeholder(dtype=tf.int32, shape=[None])
        self.obj = tf.placeholder(dtype=tf.int32, shape=[None])
        
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3])
        batch_size = tf.shape(self.images)[0]
        rgb_scaled = self.images * 255.0

        # Convert RGB to BGR
        VGG_MEAN = [103.939, 116.779, 123.68]
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        
        with tf.variable_scope("vgg19"):
            self.conv1_1 = self.conv_layer(bgr, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
            self.pool3 = self.max_pool(self.conv3_4, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
            self.pool4 = self.max_pool(self.conv4_4, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
            self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        
        shape = self.pool5.get_shape()
        size = 1
        for dim in shape[1:]:
            size *= dim.value
        
        # dense
        with tf.variable_scope('dense') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(self.pool5, [-1, size])
            weights = tf.get_variable('weights', initializer=tf.truncated_normal(shape=[size, 192]))
            biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1))
            dense = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)


        # linear layer(WX + b),
        with tf.variable_scope('softmax_linear_FA') as scope:
            weights = tf.get_variable('weights', initializer=tf.truncated_normal(shape=[192, 2]))
            biases = tf.get_variable('biases', [2], initializer=tf.constant_initializer(0.1))
            softmax_linear_FA = tf.add(tf.matmul(dense, weights), biases, name=scope.name)
            self.output_FA = tf.nn.softmax(softmax_linear_FA)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.FA, logits=softmax_linear_FA, name='cross_entropy')
            cross_entropy_mean_FA = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        with tf.variable_scope('softmax_linear_ges') as scope:
            weights = tf.get_variable('weights', initializer=tf.truncated_normal(shape=[192, 13]))
            biases = tf.get_variable('biases', [13], initializer=tf.constant_initializer(0.1))
            softmax_linear_ges = tf.add(tf.matmul(dense, weights), biases, name=scope.name)
            self.output_ges = tf.nn.softmax(softmax_linear_ges)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.ges, logits=softmax_linear_ges, name='cross_entropy')
            cross_entropy_mean_ges = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        with tf.variable_scope('softmax_linear_obj') as scope:
            weights = tf.get_variable('weights', initializer=tf.truncated_normal(shape=[192, 24]))
            biases = tf.get_variable('biases', [24], initializer=tf.constant_initializer(0.1))
            softmax_linear_obj = tf.add(tf.matmul(dense, weights), biases, name=scope.name)
            self.output_obj = tf.nn.softmax(softmax_linear_obj)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.obj, logits=softmax_linear_obj, name='cross_entropy')
            cross_entropy_mean_obj = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        self.loss = cross_entropy_mean_FA + cross_entropy_mean_ges + cross_entropy_mean_obj
        self.lr = tf.placeholder(tf.float32, [])
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
    
    def save_model(self, sess, global_step):
        var_list = [v for v in tf.global_variables() if self.name in v.name]
        saver = tf.train.Saver(var_list)
        saver.save(sess, './checkpoint/cedlCNN', global_step)
        
    def load_model(self, sess):
        var_list = [v for v in tf.global_variables() if self.name in v.name]
        saver = tf.train.Saver(var_list)
        ckpt = tf.train.get_checkpoint_state('./checkpoint/cedlCNN')
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        
tf.reset_default_graph()
width = 224
height = 224

# training parameter
batch_size = 30

# load dataset
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)

X, FA, ges, obj, X_head = load_data(os.path.join(frames_path,'train'))
X_test_path, X_test_FA, X_test_ges, X_test_obj, X_test_head_path = load_data(os.path.join(frames_path,'test'))
idx = np.random.choice(len(X), size=int(len(X)*0.3), replace=False).astype(int)
valid_idx = np.zeros(len(X), dtype=bool)
valid_idx[idx] = True
train_idx = np.ones(len(X), dtype=bool)
train_idx[idx] = False
# reduce size to debug
#debug_size = batch_size*2+1
#X = X[:debug_size]
#FA = FA[:debug_size]
#ges = ges[:debug_size]
#obj = obj[:debug_size]
#X_head = X_head[:debug_size]
#X_test_path = X_test_path[:debug_size]
#X_test_FA = X_test_FA[:debug_size]
#X_test_ges = X_test_ges[:debug_size]
#X_test_obj = X_test_obj[:debug_size]
#X_test_head_path = X_test_head_path[:debug_size]

num_train = len(X)
num_test = len(X_test_path)
print('num X_train:', num_train)
print('num X_test:',  num_test)

# shuffle training data
shuffle_idx = np.random.permutation(num_train)
X = X[shuffle_idx]
FA = FA[shuffle_idx]
ges = ges[shuffle_idx]
obj = obj[shuffle_idx]
X_head = X_head[shuffle_idx]

X_train_path = tf.constant(X[train_idx])
X_train_FA = tf.constant(FA[train_idx])
X_train_ges = tf.constant(ges[train_idx])
X_train_obj = tf.constant(obj[train_idx])
X_train_head_path = tf.constant(X_head[train_idx])
X_valid_path = tf.constant(X[valid_idx])
X_valid_FA = tf.constant(FA[valid_idx])
X_valid_ges = tf.constant(ges[valid_idx])
X_valid_obj = tf.constant(obj[valid_idx])
X_valid_head_path = tf.constant(X_head[valid_idx])
X_test_path = tf.constant(X_test_path)
X_test_FA = tf.constant(X_test_FA)
X_test_ges = tf.constant(X_test_ges)
X_test_obj = tf.constant(X_test_obj)
X_test_head_path = tf.constant(X_test_head_path)

# create TensorFlow Dataset objects
dataset = Dataset.from_tensor_slices(
    (X_train_path, X_train_FA, X_train_ges, X_train_obj, X_train_head_path))
test_dataset = Dataset.from_tensor_slices(
    (X_test_path, X_test_FA, X_test_ges, X_test_obj, X_test_head_path))
valid_dataset = Dataset.from_tensor_slices(
    (X_valid_path, X_valid_FA, X_valid_ges, X_valid_obj, X_valid_head_path))

def data_generator(X_train_path, X_train_FA, X_train_ges, X_train_obj, X_train_head_path):
    # read the img from file
    img_file = tf.read_file(X_train_path)
    img = tf.image.decode_image(img_file, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float64)
    img.set_shape([1080, 1920, 3])
    img = tf.image.resize_images(img, size=[height,width])
    img = tf.image.random_flip_left_right(img)

    # read the img from file
    img_file = tf.read_file(X_train_head_path)
    img_head = tf.image.decode_image(img_file, channels=3)
    img_head = tf.image.convert_image_dtype(img_head, tf.float64)
    img_head.set_shape([1080, 1920, 3])
    img_head = tf.image.resize_images(img_head, size=[height,width])
    img_head = tf.image.random_flip_left_right(img_head)

    return img, X_train_FA, X_train_ges, X_train_obj, img_head

dataset = dataset.map(data_generator, num_threads=4, output_buffer_size=20*batch_size)
dataset = dataset.shuffle(20*batch_size)
dataset = dataset.batch(batch_size)
test_dataset = test_dataset.map(data_generator,num_threads=4, output_buffer_size=3*batch_size)
test_dataset = test_dataset.batch(batch_size)
valid_dataset = valid_dataset.map(data_generator,num_threads=4, output_buffer_size=3*batch_size)
valid_dataset = valid_dataset.batch(batch_size)

# # create TensorFlow Iterator object
iterator = Iterator.from_structure(dataset.output_types,
                                   dataset.output_shapes)
next_element = iterator.get_next()

# # create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(dataset)
testing_init_op = iterator.make_initializer(test_dataset)
validation_init_op = iterator.make_initializer(valid_dataset)

# create model
model = Model(vgg19_npy_path='../vgg19.npy')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
valid_FA_accs = []
valid_ges_accs = []
valid_obj_accs = []
test_FA_accs = []
test_ges_accs = []
test_obj_accs = []
train_losses = []
test_losses = []
valid_losses = []
best_acc = 0

def learning_rate(epoch):
    return min(0.0001, 0.001*(0.95**epoch))

for epoch in range(50):
    sess.run(training_init_op)
    total = 0
    FA_acc = 0
    ges_acc = 0
    obj_acc = 0
    losses = 0
    while True:
        try:
            X_train, X_train_FA, X_train_ges, X_train_obj, X_train_head = sess.run(next_element)
            step = sess.run(model.global_step)
            feed_dict = {
                model.images: X_train,
                model.FA: X_train_FA,
                model.ges: X_train_ges,
                model.obj: X_train_obj,
                model.lr: learning_rate(epoch),
            }
            _, loss, step, output_FA, output_ges, output_obj = sess.run(
                [model.train_op, model.loss, model.global_step, model.output_FA,
                 model.output_ges, model.output_obj], feed_dict=feed_dict)
            
            FA_acc += np.sum(np.argmax(output_FA, axis=1)==X_train_FA)
            ges_acc += np.sum(np.argmax(output_ges, axis=1)==X_train_ges)
            obj_acc += np.sum(np.argmax(output_obj, axis=1)==X_train_obj)
            losses += loss*len(X_train)
            total += len(X_train)
            
        except tf.errors.OutOfRangeError:
            losses /= total
            train_losses.append(losses)
            FA_acc = FA_acc/total
            ges_acc = ges_acc/total
            obj_acc = obj_acc/total
            print('INFO:root:Epoch[%d] loss=%f' %(epoch, losses))
            print('INFO:root:Epoch[%d] FA training-accuracy=%f' %(epoch, FA_acc))
            print('INFO:root:Epoch[%d] ges training-accuracy=%f' %(epoch, ges_acc))
            print('INFO:root:Epoch[%d] obj training-accuracy=%f' %(epoch, obj_acc))
            print()
            break
            
    sess.run(validation_init_op)
    total = 0
    FA_acc = 0
    ges_acc = 0
    obj_acc = 0
    losses = 0
    while True:
        try:
            X_train, X_train_FA, X_train_ges, X_train_obj, X_train_head = sess.run(next_element)
            feed_dict = {
                model.images: X_train,
                model.FA: X_train_FA,
                model.ges: X_train_ges,
                model.obj: X_train_obj,
            }
            loss, output_FA, output_ges, output_obj, step = sess.run(
                    [model.loss, model.output_FA, model.output_ges, model.output_obj, model.global_step], feed_dict=feed_dict)
            
            FA_acc += np.sum(np.argmax(output_FA, axis=1)==X_train_FA)
            ges_acc += np.sum(np.argmax(output_ges, axis=1)==X_train_ges)
            obj_acc += np.sum(np.argmax(output_obj, axis=1)==X_train_obj)
            losses += loss*len(X_train)
            total += len(X_train)
            
        except tf.errors.OutOfRangeError:
            losses /= total
            valid_losses.append(losses)
            FA_acc = FA_acc/total
            ges_acc = ges_acc/total
            obj_acc = obj_acc/total
            valid_FA_accs.append(FA_acc)
            valid_ges_accs.append(ges_acc)
            valid_obj_accs.append(obj_acc)
            model.save_model(sess, epoch)
            
            print('INFO:root:Epoch[%d] loss=%f' %(epoch, losses))
            print('INFO:root:Epoch[%d] FA validation-accuracy=%f' %(epoch, FA_acc))
            print('INFO:root:Epoch[%d] ges validation-accuracy=%f' %(epoch, ges_acc))
            print('INFO:root:Epoch[%d] obj validation-accuracy=%f' %(epoch, obj_acc))
            print()
            break
            
    sess.run(testing_init_op)
    total = 0
    FA_acc = 0
    ges_acc = 0
    obj_acc = 0
    losses = 0
    while True:
        try:
            X_train, X_train_FA, X_train_ges, X_train_obj, X_train_head = sess.run(next_element)
            feed_dict = {
                model.images: X_train,
                model.FA: X_train_FA,
                model.ges: X_train_ges,
                model.obj: X_train_obj,
            }
            loss, output_FA, output_ges, output_obj, step = sess.run(
                    [model.loss, model.output_FA, model.output_ges, model.output_obj, model.global_step], feed_dict=feed_dict)
            
            FA_acc += np.sum(np.argmax(output_FA, axis=1)==X_train_FA)
            ges_acc += np.sum(np.argmax(output_ges, axis=1)==X_train_ges)
            obj_acc += np.sum(np.argmax(output_obj, axis=1)==X_train_obj)
            losses += loss*len(X_train)
            total += len(X_train)
            
        except tf.errors.OutOfRangeError:
            losses /= total
            test_losses.append(losses)
            FA_acc = FA_acc/total
            ges_acc = ges_acc/total
            obj_acc = obj_acc/total
            test_FA_accs.append(FA_acc)
            test_ges_accs.append(ges_acc)
            test_obj_accs.append(obj_acc)
            
            print('INFO:root:Epoch[%d] loss=%f' %(epoch, losses))
            print('INFO:root:Epoch[%d] FA testing-accuracy=%f' %(epoch, FA_acc))
            print('INFO:root:Epoch[%d] ges testing-accuracy=%f' %(epoch, ges_acc))
            print('INFO:root:Epoch[%d] obj testing-accuracy=%f' %(epoch, obj_acc))
            print()
            break
            
valid_FA_accs = np.array(valid_FA_accs)
valid_ges_accs = np.array(valid_ges_accs)
valid_obj_asss = np.array(valid_obj_accs)
test_FA_accs = np.array(test_FA_accs)
test_ges_accs = np.array(test_ges_accs)
test_obj_asss = np.array(test_obj_accs)
test_losses = np.array(test_losses)
np.save('test_FA_acc.npy', test_FA_accs)
np.save('test_ges_acc.npy', test_ges_accs)
np.save('test_obj_acc.npy', test_obj_accs)
np.save('valid_FA_acc.npy', valid_FA_accs)
np.save('valid_ges_acc.npy', valid_ges_accs)
np.save('valid_obj_acc.npy', valid_obj_accs)
np.save('test_loss.npy', test_losses)
np.save('train_loss.npy', train_losses)
np.save('valid_loss.npy', valid_loss)