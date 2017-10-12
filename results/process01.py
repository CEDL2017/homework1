import numpy as np
import os
import tensorflow as tf

env = os.environ.get('GRAPE_DATASET_DIR')
testData = np.load(os.path.join(env,"testData.npy"))
trainData = np.load(os.path.join(env,"trainData.npy"))
testLabel = np.load(os.path.join(env,"testLabel.npy"))
trainLabel = np.load(os.path.join(env,"trainLabel.npy"))

learning_rate = 0.001
training_iters = 200

n_input = 45*80*3
n_classes = 24
dropout = 0.8

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
    _X = tf.reshape(_X, shape=[-1, 45, 80, 3])

    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    pool1 = max_pool('pool1', conv1, k=2)
    norm1 = norm('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, _dropout)

    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    pool2 = max_pool('pool2', conv2, k=2)
    norm2 = norm('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, _dropout)

    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    pool3 = max_pool('pool3', conv3, k=2)
    norm3 = norm('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, _dropout)

    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')

    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([6*10*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 24]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = alex_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

N = 200

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    Ist = 0
    Ien = Ist+N
    while step  < training_iters:
        if Ien > trainData.shape[0]:
            Ien = trainData.shape[0]
        sess.run(optimizer, feed_dict={x: trainData[Ist:Ien], y: trainLabel[Ist:Ien], keep_prob: dropout})
        Ist+=N
        Ien+=N
        if Ist > trainData.shape[0]:
            Ist = 0
            Ien = Ist+N
            ist = 0
            ien = ist+N
            temp = 0
            while ist < trainData.shape[0]:
                if ien > trainData.shape[0]:
                    ien = trainData.shape[0]
                temp += sess.run(accuracy, feed_dict={x: trainData[ist:ien], y: trainLabel[ist:ien], keep_prob: 1.})
                ist+=N
                ien+=N
            acc = temp/trainData.shape[0]
            ist = 0
            ien = ist+N
            temp = 0
            while ist < testData.shape[0]:
                if ien > testData.shape[0]:
                    ien = testData.shape[0]
                temp += sess.run(accuracy, feed_dict={x: testData[ist:ien], y: testLabel[ist:ien], keep_prob: 1.})
                ist+=N
                ien+=N
            v_acc = temp/testData.shape[0] 
            print('\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %(step, acc, step, v_acc))
            step += 1