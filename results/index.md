# 江愷笙 <span style="color:red">(106062568)</span>

#Project 5: Deep Classification

## Overview
The project is related to 
* Cheng-Sheng Chan, Shou-Zhong Chen, Pei-Xuan Xie, Chiung-Chih Chang, Min Sun, "Recognition from Hand Cameras: A Revisit with Deep Learning." ECCV 2016
* Alex Krizhevsky, Ilya Sutskever, Geoffery E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks"
* Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition"
在本次作業中我們需要將提供的training image, training label讀入tensorflow中，訓練完後再利用testing data做測試，而這次的資料有分為左右手camera的資料以及頭部camera的資料，
透過這些camera可以讓機器針對使用者日常接觸的物體作分類以及判斷，不過因為頭部的資料是沒有label的，在本次作業中我只讀取左右手的資料來做訓練以及測試

## Implementation

1. input data
   在input_data.py這個檔案裡我分為兩部分，一個是讀取image和label並將他們配對，使得image能對應到正確的label，另一個則是將這些data分成一個一個的batch，以方便我們做training
   * read_data():
   function:
   在function中我定義傳入的為image所在的路徑以及label的路徑，且若train為True的話，我們讀取的是training的image和label，而若train為False的話，則讀取testing的image和label
   ```python
   def read_data(file_dir, label_dir, train = True):
   ```
   for image:
   ```python
    image_list = []
    image_temp = []
    for root, direc, files in os.walk(file_dir):						# use os.walk to go through each folder and file
        for name in files:
            image_temp.append(os.path.join(root, name))						# add each file which in the current folder to a temp list
	image_list = image_list + sorted(image_temp, key = lambda x: int(re.sub('\D', '', x)))	# sort the temp list and add it to the original image list
        image_temp = []   
   ```
   for label:

   ```python
    label_list = []  
    for root, direc, files in os.walk(label_dir):
        if train:   										# train label for lab is 1234
            if root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/lab':
                for i in range(1, 5):
                    label_list.extend(np.load(os.path.join(root, 'obj_left'+ str(i) + '.npy')))
                for i in range(1, 5):
                    label_list.extend(np.load(os.path.join(root, 'obj_right'+ str(i) + '.npy')))
            if (root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/house') or (root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/office'):
                for i in range(1, 4):
                    label_list.extend(np.load(os.path.join(root, 'obj_left'+ str(i) + '.npy')))
                for i in range(1, 4):
                    label_list.extend(np.load(os.path.join(root, 'obj_right'+ str(i) + '.npy')))
        else:											# test label for lab is 5678
            if root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/lab':
                for i in range(5, 9):
                    label_list.extend(np.load(os.path.join(root, 'obj_left'+ str(i) + '.npy')))
                for i in range(5, 9):
                    label_list.extend(np.load(os.path.join(root, 'obj_right'+ str(i) + '.npy')))
            if(root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/house') or (root == '/home/viplab/Desktop/petersci/CEDL_HW1/data/labels/office'):
                for i in range(4, 7):
                    label_list.extend(np.load(os.path.join(root, 'obj_left'+ str(i) + '.npy')))
                for i in range(4, 7):
                    label_list.extend(np.load(os.path.join(root, 'obj_right'+ str(i) + '.npy')))
   ```
   接著我們再利用np.random.shuffle()來將配好的image和label打亂，因為batch中如果圖片較隨機的話就不會讓機器在同一個batch一直看到類似的圖片，train出來的結果會比較真實

   * batch_generate():
   而在generate batch時我使用一個queue來存取batch，這樣每次取出來的都會是不同的資料，並且在這邊做image的resize，
   因為原本1920x1080的大小太大了，所以先做resize之後再丟到network裡面做訓練，最後需要將其轉換成one hot的型態以方便進行classification   
   
   ```python
    label_batch = tf.one_hot(label_batch, 24)
    label_batch = tf.cast(label_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size, 24])
    image_batch = tf.cast(image_batch, tf.float32)

   ```

2. Alexnet model 
   根據原本的Alexnet paper，他們每層捲積核的數目分別為96, 256.384,384,256，兩層fully connected layer的neuron數目各為4096，最後輸出1000種類別，
   由於我們這次需要便是的只有object，且只有24種類別，因此我將輸出由1000改為24，而每層的捲積和數目也變淺一些，而在fully connected layer的neuron樹則各為1024，
   一方面因為我們要分辨的種類比較少。另一方面也是受限於GPU的記憶體大小，常常架構寫得太大就會跳出記憶體不足的問題
   
   ```python
    # Alexnet structure
    # conv_layer1
    conv1 = tf.layers.conv2d(xs, filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu)
    MaxPool1 = tf.layers.max_pooling2d(conv1, pool_size = 2, strides = 2)
    norm1 = tf.nn.lrn(MaxPool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

    # conv_layer2
    conv2 = tf.layers.conv2d(norm1, 256, 3, 1, 'same', activation = tf.nn.relu)
    MaxPool2 = tf.layers.max_pooling2d(conv2, pool_size = 2, strides = 2)
    norm2 = tf.nn.lrn(MaxPool2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

    # conv_layer3
    conv3 = tf.layers.conv2d(norm2, 300, 3, 1, 'same', activation = tf.nn.relu)
    #MaxPool3 = tf.layers.max_pooling2d(conv3, pool_size = 2, strides = 2)

    # conv_layer4
    conv4 = tf.layers.conv2d(conv3, 300, 3, 1, 'same', activation = tf.nn.relu)
    #MaxPool4 = tf.layers.max_pooling2d(conv4, pool_size = 2, strides = 2)

    # conv_layer5
    conv5 = tf.layers.conv2d(conv4, 256, 3, 1, 'same', activation = tf.nn.relu)
    MaxPool5 = tf.layers.max_pooling2d(conv5, pool_size = 2, strides = 2)
    #norm5 = tf.nn.lrn(MaxPool5, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

    # flatened
    flat = tf.reshape(MaxPool5, [-1, 25*25*256])

    # fully connected layer6
    fc6 = tf.layers.dense(flat, 1024, activation = tf.nn.relu)
    drop6 = tf.nn.dropout(fc6, dropout)

    # fully connected layer7
    fc7 = tf.layers.dense(drop6, 1024, activation = tf.nn.relu)
    drop7 = tf.nn.dropout(fc7, dropout)

    # fully connected layer8
    output = tf.layers.dense(drop7, 24, activation = tf.nn.softmax)

   ```

3. VGG19 model
   VGG19的網路架構與Alexnet非常像，只是在convolution layer裡多做了幾次convolution，同時因為做norm會降低training的效率，所以將norm給拿掉，
   同樣因為VGG用來分辨1000種物體，而我們只須分別24種，因此在輸出的地方須做更改，同時我也減小了每個convolution的深度以及最後fc層的大小
   
   ```python
   # VGG structure
   # conv_layer1
   conv1_1 = tf.layers.conv2d(xs, filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu)
   conv1_2 = tf.layers.conv2d(conv1_1, 32, 3, 1, 'same', activation = tf.nn.relu)
   MaxPool1 = tf.layers.max_pooling2d(conv1_2, pool_size = 2, strides = 2)
   #norm1 = tf.nn.lrn(MaxPool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

   # conv_layer2
   conv2_1 = tf.layers.conv2d(MaxPool1, 64, 3, 1, 'same', activation = tf.nn.relu)
   conv2_2 = tf.layers.conv2d(conv2_1, 64, 3, 1, 'same', activation = tf.nn.relu)
   MaxPool2 = tf.layers.max_pooling2d(conv2_2, pool_size = 2, strides = 2)
   #norm2 = tf.nn.lrn(MaxPool2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

   # conv_layer3
   conv3_1 = tf.layers.conv2d(MaxPool2, 128, 3, 1, 'same', activation = tf.nn.relu)
   conv3_2 = tf.layers.conv2d(conv3_1, 128, 3, 1, 'same', activation = tf.nn.relu)
   conv3_3 = tf.layers.conv2d(conv3_2, 128, 3, 1, 'same', activation = tf.nn.relu)
   conv3_4 = tf.layers.conv2d(conv3_3, 128, 3, 1, 'same', activation = tf.nn.relu)
   MaxPool3 = tf.layers.max_pooling2d(conv3_4, pool_size = 2, strides = 2)

   # conv_layer4
   conv4_1 = tf.layers.conv2d(MaxPool3, 256, 3, 1, 'same', activation = tf.nn.relu)
   conv4_2 = tf.layers.conv2d(conv4_1, 256, 3, 1, 'same', activation = tf.nn.relu)
   conv4_3 = tf.layers.conv2d(conv4_2, 256, 3, 1, 'same', activation = tf.nn.relu)
   conv4_4 = tf.layers.conv2d(conv4_3, 256, 3, 1, 'same', activation = tf.nn.relu)
   MaxPool4 = tf.layers.max_pooling2d(conv4_4, pool_size = 2, strides = 2)

   # conv_layer5
   conv5_1 = tf.layers.conv2d(MaxPool4, 256, 3, 1, 'same', activation = tf.nn.relu)
   conv5_2 = tf.layers.conv2d(conv5_1, 256, 3, 1, 'same', activation = tf.nn.relu)
   conv5_3 = tf.layers.conv2d(conv5_2, 256, 3, 1, 'same', activation = tf.nn.relu)
   conv5_4 = tf.layers.conv2d(conv5_3, 256, 3, 1, 'same', activation = tf.nn.relu)
   MaxPool5 = tf.layers.max_pooling2d(conv5_4, pool_size = 2, strides = 2)

   # flatened
   flat = tf.reshape(MaxPool5, [-1, 7*7*256])

   # fully connected layer6
   fc6 = tf.layers.dense(flat, 1024, activation = tf.nn.relu)
   drop6 = tf.nn.dropout(fc6, dropout)

   # fully connected layer7
   fc7 = tf.layers.dense(drop6, 1024, activation = tf.nn.relu)
   drop7 = tf.nn.dropout(fc7, dropout)

   # fully connected layer8
   output = tf.layers.dense(drop7, 24)
   ```

4. train and test
   在tensorflow裡面我們需要開session來跑我們要訓練的網路，在此處我將iteration的數量定為總image數除以batch size，因為不一定整除，所以還須將其轉換為int，
   而訓練方式是每一次都將一個batch丟進network做訓練，之後換下一個batch，直到跑完所有的batch後，就跑完一個epoch，在本次作業中我設定epoch數目為20，
   並在每一個epoch之後print出這個epoch平均的loss以及training的accuracy，等跑完20個epoch後再將testing的資料丟進去計算test的accuracy

   ```python
   saver = tf.train.Saver()

   # start a session
   sess = tf.Session()
   init = tf.global_variables_initializer()
   sess.run(init)                                          # initialize all global variables
   sess.run(tf.local_variables_initializer())              # initialize all local variables(such as acc_op)

   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess=sess, coord=coord)

   tot_count = 0
   count = 0
   compute_accuracy = 0
   tot_accuracy = 0
   epoch_loss = 0
   tot_loss = 0
   for i in range(epoch):
       print('[epoch %d]' % (i+1))
       for iteration in range(int(N_sample / batch_size)):
           if coord.should_stop():
               break
           count += 1
           tot_count += batch_size 
           image_batch, label_batch = sess.run([batch_xs, batch_ys])
           loss_step = sess.run(loss, feed_dict = {xs: image_batch, ys: label_batch})
           train_step = sess.run(train, feed_dict = {xs: image_batch, ys: label_batch})
           acc_correct = sess.run(accuracy, feed_dict = {xs: image_batch, ys: label_batch})
           compute_accuracy = acc_correct + compute_accuracy
           tot_accuracy = compute_accuracy / tot_count
           epoch_loss = loss_step + epoch_loss
           tot_loss = epoch_loss / count
           if iteration % 50 == 0:
               print('Step:', count, '| loss: %.4f' % loss_step, '| train accuracy: %.4f' % tot_accuracy)
           if (i == epoch - 1) and (iteration == int(N_sample / batch_size) - 1):
               checkpoint_path = os.path.join(save_dir, 'model.ckpt')
               saver.save(sess, checkpoint_path, global_step=count)
       print('[epoch %d] ======> ' % (i+1), 'epoch loss: %.4f' % tot_loss, 'epoch accuracy: %.4f' % tot_accuracy)

   coord.request_stop()       
   coord.join(threads)
   sess.close()
   ```

## Installation
* Tensorflow
* Python3
* use command `python3 VGG19.py` to train the neural network

## Results
在此處比較Alexnet以及VGG19兩個網路架構的結果，由下圖可以看出經由Alexnet訓練之後的正確率為50%，然而可以發現training data的loss卻沒有顯著的下降，
因為在train以及testing的效果都不是很好，原先想法是可能是underfitting，因此將原本的Alexnet換成VGG19的網路架構，

<center>
<img src="pictures/Alex_test.png" alt="Alex_test" style="float:middle;">
</center>

替換成VGG19後的結果如下圖所示，後來想到也有可能是網路太深使得產生gradient vanishing的問題，所以在VGG19的地方將每個convolution的深度做了調整，將它們變淺
但看起來效果並不是特別明顯，之後可能會採用ResNet來測試是否能改善這樣的問題

<center>
<img src="picture/VGG_epoch1.png" alt="VGG_epoch1" style="float:middle;">
</center>

<center>
<img src="picture/VGG_epoch2.png" alt="VGG_epoch1" style="float:middle;">
</center>

<center>
<img src="picture/VGG_epoch20.png" alt="VGG_epoch1" style="float:middle;">
</center>

### test accuracy
* AlexNet: 50%
* VGG19: 50.08%

