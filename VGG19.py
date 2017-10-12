import tensorflow as tf

#%%
def conv (layer_name, image, out_channels, kernel_size=[3,3], stride=[1,1,1,1]):
    '''layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]'''
   
    in_channels = image.get_shape()[-1]    
    
    with tf.variable_scope(layer_name):
        Weights = tf.get_variable(name='weights',
                                  shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                  dtype = tf.float32 )
    
        biases = tf.get_variable(name='bias', 
                                 shape=[out_channels],
                                 dtype=tf.float32 )
    
        conv = tf.nn.conv2d(image, Weights, stride, padding='SAME',name='conv')
        pre_activation = tf.nn.bias_add(conv, biases)
        pre_activation = tf.reshape(pre_activation, conv.get_shape().as_list()) #get_shape return的value是tensorshape (位元組)，用as_list()得到具體尺寸
        conv = tf.nn.relu(pre_activation)
        return conv
#%%
def Max_pool(layer_name, conv, kernel, stride):

    conv = tf.nn.max_pool(conv, kernel, strides=stride, padding='SAME', name=layer_name)

    return conv
#%%
def FC_layer(layer_name, input_Feature, out_nodes , is_relu):

    shape = input_Feature.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        Weights = tf.get_variable('w',
                                shape=[size, out_nodes],
                                dtype=tf.float32 )

        biases = tf.get_variable('b',
                                shape=[out_nodes],
                                dtype=tf.float32 )
        flat_x = tf.reshape(input_Feature, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, Weights), biases)
        if is_relu :
            x = tf.nn.relu(x)

        return x
#%%   
def VGG19(images,  n_classes):
    x_feature = conv('conv1_1', images, 8, kernel_size=[3,3], stride=[1,1,1,1])   
    x_feature = conv('conv1_2', x_feature, 8, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = Max_pool('pool1', x_feature, kernel=[1,2,2,1], stride=[1,2,2,1])
    
    x_feature = conv('conv2_1', x_feature, 16, kernel_size=[3,3], stride=[1,1,1,1])    
    x_feature = conv('conv2_2', x_feature, 16, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = Max_pool('pool2', x_feature, kernel=[1,2,2,1], stride=[1,2,2,1])
         
            
    x_feature = conv('conv3_1', x_feature, 32, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = conv('conv3_2', x_feature, 32, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = conv('conv3_3', x_feature, 32, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = conv('conv3_4', x_feature, 32, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = Max_pool('pool3', x_feature, kernel=[1,2,2,1], stride=[1,2,2,1])
            

    x_feature = conv('conv4_1', x_feature, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = conv('conv4_2', x_feature, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = conv('conv4_3', x_feature, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = conv('conv4_4', x_feature, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = Max_pool('pool4', x_feature, kernel=[1,2,2,1], stride=[1,2,2,1])
        
    x_feature = conv('conv5_1', x_feature, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = conv('conv5_2', x_feature, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = conv('conv5_3', x_feature, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = conv('conv5_4', x_feature, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x_feature = Max_pool('pool5', x_feature, kernel=[1,2,2,1], stride=[1,2,2,1])            
        
        
    x_feature = FC_layer('fc6', x_feature, out_nodes=512, is_relu = True)        
    x_feature = tf.nn.dropout(x_feature, 0.5)    

    x_feature = FC_layer('fc7', x_feature, out_nodes=512, is_relu = True)        
    x_feature = tf.nn.dropout(x_feature, 0.5)

    x_feature = FC_layer('fc8', x_feature, out_nodes=n_classes, is_relu = False)
    x_feature = tf.nn.softmax(x_feature)

   
    return x_feature



#%%
def losses(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels [batch_size, n_classes]
    '''
    with tf.name_scope('loss') as scope:
        #corss entropy 是loss函數的一種，描述predictionValue 與real value差多少
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        #loss = tf.reduce_sum(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss

#%%
def trainning(loss, learning_rate):

    with tf.variable_scope('optimizer'):

        #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        optimizer = tf.contrib.opt.NadamOptimizer().minimize(loss)
    return optimizer
#%%
def evaACC(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, float - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  with tf.name_scope('accuracy') as scope:
      correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))#為bool表
      correct = tf.cast(correct, tf.float32)#把bool表轉成[1,0,0,1...]方便計算
      accuracy = tf.reduce_mean(correct) #獲得準確率
      tf.summary.scalar(scope+'/accuracy', accuracy) #tensorflow視覺化的東西
  return accuracy

def prediction_acc(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Return:
      the number of correct predictions
  """
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  correct = tf.cast(correct, tf.float32)
  n_correct = tf.reduce_sum(correct)
  return n_correct



