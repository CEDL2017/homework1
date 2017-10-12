# DL_HW_1

106062623 楊朝勛

#Project 1: Deep Classification

Overview
---
Recently, the technological advance of wearable devices has led to significant interests in recognizing human behaviors in daily life (i.e., uninstrumented environment). Among many devices, egocentric camera systems have drawn significant attention, since the camera is aligned with the field-of-view of wearer, it naturally captures what a person sees. These systems have shown great potential in recognizing daily activities(e.g., making meals, watching TV, etc.), estimating hand poses, generating howto videos, etc.

Despite many advantages of egocentric camera systems, there exists two main issues which are much less discussed. Firstly, hand localization is not solved especially for passive camera systems. Even for active camera systems like Kinect, hand localization is challenging when two hands are interacting or a hand is interacting with an object. Secondly, the limited field-of-view of an egocentric camera implies that hands will inevitably move outside the images sometimes.

HandCam (Fig. 1), a novel wearable camera capturing activities of hands, for recognizing human behaviors. HandCam has two main advantages over egocentric systems : (1) it avoids the need to detect hands and manipulation regions; (2) it observes the activities of hands almost at all time.


Implementation
---
<br/>
    In the implementation of model, there're two structure we have provided: simplified VGG19 and simplified VGG16. You can set the use_VGG flag (in lib/config.py)to switch the structure.
    
    ```
    Obj = { 'free':0,
        'computer':1,
        'cellphone':2,
        'coin':3,
        'ruler':4,
        'thermos-bottle':5,
        'whiteboard-pen':6,
        'whiteboard-eraser':7,
        'pen':8,
        'cup':9,
        'remote-control-TV':10,
        'remote-control-AC':11,
        'switch':12,
        'windows':13,
        'fridge':14,
        'cupboard':15,
        'water-tap':16,
        'toy':17,
        'kettle':18,
        'bottle':19,
        'cookie':20,
        'book':21,
        'magnet':22,
        'lamp-switch':23}
      ```

Code highlights
            VGG 19:
            ```
            self.model.add(Conv2D(8, (3, 3), padding='same', name="conv1_1", activation="relu",input_shape=(img_height, img_width, 3)))
            
            self.model.add(Conv2D(8, (3, 3), padding='same', name="conv1_2", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            
            self.model.add(Conv2D(16, (3, 3), padding='same',name="conv2_1", activation="relu"))
            self.model.add(Conv2D(16, (3, 3), padding='same', name="conv2_2", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_1", activation="relu"))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_2", activation="relu"))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_3", activation="relu"))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_4", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_1", activation="relu"))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_2", activation="relu"))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_3", activation="relu"))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_4", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_1", activation="relu"))
   
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_2", activation="relu"))

            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_3", activation="relu"))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_4", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            self.model.add(Flatten(name="flatten"))
            self.model.add(Dense(512, activation='relu', name='dense_1'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(512, activation='relu', name='dense_2'))
            self.model.add(Dropout(0.5))

            self.model.add(Dense(scorenet_fc_num,activation='softmax',name='final_layer'))
            ```
Installation
---
env setting:
if python 2.X use pip to install
if python 3.X use pip3 to install

using keras(tensorflow backend):
 ```
            pip install tensorflow
            pip install keras
            pip install scikit-image
            pip install matplotlib
            pip install Pillow
 ```
            
Train the  model:
    if you want to train a whole new model remenber to remove `scorenet.h5` in model folder
    
    ```
    $ cd train
    $ python scorenet_train.py
    ```

Test the  model:

    ```
    $ cd test
    $ python Accuracy_test.py
    ```

name rule of pre-trained model:
`img height x img width_NumOfIter(Accurancy)`(eg.`150x150_200iter(0.478).h5`)（i.e. NumOfIter:iterator次數,假設圖片有14992張
,每14992為一個iteraor）
or `img height x img width_NumOfFreeLabelImg(Accurancy)`(eg.`200x200_free200(0.348).h5`)

the model will save in model folder(initialize name is`scorenet.h5`)
if you want to test another pre-trained model ,you can set the model_path (in lib/config.py)to switch the structure.
eg. if you want to load model which name'200x200_200iter(0.478)'

    ```
    img_height = 200
    img_width = 200
    .
    .
    .
    model_path = '../model/150x150_200iter(0.478).h5'
    ```
    
data支路敬請修改config.py的以下code

```
scorenet_img_path_house1L = os.path.join('/media/timyang/My Passport/frames/train/house/1/Lhand/')
scorenet_img_path_house1R = os.path.join('/media/timyang/My Passport/frames/train/house/1/Rhand/')
scorenet_img_path_house2L = os.path.join('/media/timyang/My Passport/frames/train/house/2/Lhand/')
scorenet_img_path_house2R = os.path.join('/media/timyang/My Passport/frames/train/house/2/Rhand/')
scorenet_img_path_house3L = os.path.join('/media/timyang/My Passport/frames/train/house/3/Lhand/')
scorenet_img_path_house3R = os.path.join('/media/timyang/My Passport/frames/train/house/3/Rhand/')
scorenet_img_path_lab1L= os.path.join('/media/timyang/My Passport/frames/train/lab/1/Lhand/')
scorenet_img_path_lab1R= os.path.join('/media/timyang/My Passport/frames/train/lab/1/Rhand/')
scorenet_img_path_lab2L= os.path.join('/media/timyang/My Passport/frames/train/lab/2/Lhand/')
scorenet_img_path_lab2R= os.path.join('/media/timyang/My Passport/frames/train/lab/2/Rhand/')
scorenet_img_path_lab3L= os.path.join('/media/timyang/My Passport/frames/train/lab/3/Lhand/')
scorenet_img_path_lab3R= os.path.join('/media/timyang/My Passport/frames/train/lab/3/Rhand/')
scorenet_img_path_lab4L= os.path.join('/media/timyang/My Passport/frames/train/lab/4/Lhand/')
scorenet_img_path_lab4R= os.path.join('/media/timyang/My Passport/frames/train/lab/4/Rhand/')
scorenet_img_path_office1L = os.path.join('/media/timyang/My Passport/frames/train/office/1/Lhand/')
scorenet_img_path_office1R = os.path.join('/media/timyang/My Passport/frames/train/office/1/Rhand/')
scorenet_img_path_office2L = os.path.join('/media/timyang/My Passport/frames/train/office/2/Lhand/')
scorenet_img_path_office2R = os.path.join('/media/timyang/My Passport/frames/train/office/2/Rhand/')
scorenet_img_path_office3L = os.path.join('/media/timyang/My Passport/frames/train/office/3/Lhand/')
scorenet_img_path_office3R = os.path.join('/media/timyang/My Passport/frames/train/office/3/Rhand/')

train_img_path_house1L = os.path.join('/media/timyang/My Passport/frames/test/house/1/Lhand/')
train_img_path_house1R = os.path.join('/media/timyang/My Passport/frames/test/house/1/Rhand/')
train_img_path_house2L = os.path.join('/media/timyang/My Passport/frames/test/house/2/Lhand/')
train_img_path_house2R = os.path.join('/media/timyang/My Passport/frames/test/house/2/Rhand/')
train_img_path_house3L = os.path.join('/media/timyang/My Passport/frames/test/house/3/Lhand/')
train_img_path_house3R = os.path.join('/media/timyang/My Passport/frames/test/house/3/Rhand/')
train_img_path_lab1L= os.path.join('/media/timyang/My Passport/frames/test/lab/1/Lhand/')
train_img_path_lab1R= os.path.join('/media/timyang/My Passport/frames/test/lab/1/Rhand/')
train_img_path_lab2L= os.path.join('/media/timyang/My Passport/frames/test/lab/2/Lhand/')
train_img_path_lab2R= os.path.join('/media/timyang/My Passport/frames/test/lab/2/Rhand/')
train_img_path_lab3L= os.path.join('/media/timyang/My Passport/frames/test/lab/3/Lhand/')
train_img_path_lab3R= os.path.join('/media/timyang/My Passport/frames/test/lab/3/Rhand/')
train_img_path_lab4L= os.path.join('/media/timyang/My Passport/frames/test/lab/4/Lhand/')
train_img_path_lab4R= os.path.join('/media/timyang/My Passport/frames/test/lab/4/Rhand/')
train_img_path_office1L = os.path.join('/media/timyang/My Passport/frames/test/office/1/Lhand/')
train_img_path_office1R = os.path.join('/media/timyang/My Passport/frames/test/office/1/Rhand/')
train_img_path_office2L = os.path.join('/media/timyang/My Passport/frames/test/office/2/Lhand/')
train_img_path_office2R = os.path.join('/media/timyang/My Passport/frames/test/office/2/Rhand/')
train_img_path_office3L = os.path.join('/media/timyang/My Passport/frames/test/office/3/Lhand/')
train_img_path_office3R = os.path.join('/media/timyang/My Passport/frames/test/office/3/Rhand/')


# scorenet_dat_path_zip = os.path.join(data_env,'labels.zip')
scorenet_dat_path_house1L = os.path.join('/media/timyang/My Passport/labels/house/obj_left1.npy')
scorenet_dat_path_house1R = os.path.join('/media/timyang/My Passport/labels/house/obj_right1.npy')
scorenet_dat_path_house2L= os.path.join('/media/timyang/My Passport/labels/house/obj_left2.npy')
scorenet_dat_path_house2R= os.path.join('/media/timyang/My Passport/labels/house/obj_right2.npy')
scorenet_dat_path_house3L= os.path.join('/media/timyang/My Passport/labels/house/obj_left3.npy')
scorenet_dat_path_house3R= os.path.join('/media/timyang/My Passport/labels/house/obj_right3.npy')
scorenet_dat_path_lab1L= os.path.join('/media/timyang/My Passport/labels/lab/obj_left1.npy')
scorenet_dat_path_lab1R= os.path.join('/media/timyang/My Passport/labels/lab/obj_right1.npy')
scorenet_dat_path_lab2L= os.path.join('/media/timyang/My Passport/labels/lab/obj_left2.npy')
scorenet_dat_path_lab2R= os.path.join('/media/timyang/My Passport/labels/lab/obj_right2.npy')
scorenet_dat_path_lab3L= os.path.join('/media/timyang/My Passport/labels/lab/obj_left3.npy')
scorenet_dat_path_lab3R= os.path.join('/media/timyang/My Passport/labels/lab/obj_right3.npy')
scorenet_dat_path_lab4L= os.path.join('/media/timyang/My Passport/labels/lab/obj_left4.npy')
scorenet_dat_path_lab4R= os.path.join('/media/timyang/My Passport/labels/lab/obj_right4.npy')
scorenet_dat_path_office1L= os.path.join('/media/timyang/My Passport/labels/office/obj_left1.npy')
scorenet_dat_path_office1R= os.path.join('/media/timyang/My Passport/labels/office/obj_right1.npy')
scorenet_dat_path_office2L= os.path.join('/media/timyang/My Passport/labels/office/obj_left2.npy')
scorenet_dat_path_office2R= os.path.join('/media/timyang/My Passport/labels/office/obj_right2.npy')
scorenet_dat_path_office3L= os.path.join('/media/timyang/My Passport/labels/office/obj_left3.npy')
scorenet_dat_path_office3R= os.path.join('/media/timyang/My Passport/labels/office/obj_right3.npy')

train_dat_path_house1L = os.path.join('/media/timyang/My Passport/labels/house/obj_left4.npy')
train_dat_path_house1R = os.path.join('/media/timyang/My Passport/labels/house/obj_right4.npy')
train_dat_path_house2L= os.path.join('/media/timyang/My Passport/labels/house/obj_left5.npy')
train_dat_path_house2R= os.path.join('/media/timyang/My Passport/labels/house/obj_right5.npy')
train_dat_path_house3L= os.path.join('/media/timyang/My Passport/labels/house/obj_left6.npy')
train_dat_path_house3R= os.path.join('/media/timyang/My Passport/labels/house/obj_right6.npy')
train_dat_path_lab1L= os.path.join('/media/timyang/My Passport/labels/lab/obj_left5.npy')
train_dat_path_lab1R= os.path.join('/media/timyang/My Passport/labels/lab/obj_right5.npy')
train_dat_path_lab2L= os.path.join('/media/timyang/My Passport/labels/lab/obj_left6.npy')
train_dat_path_lab2R= os.path.join('/media/timyang/My Passport/labels/lab/obj_right6.npy')
train_dat_path_lab3L= os.path.join('/media/timyang/My Passport/labels/lab/obj_left7.npy')
train_dat_path_lab3R= os.path.join('/media/timyang/My Passport/labels/lab/obj_right7.npy')
train_dat_path_lab4L= os.path.join('/media/timyang/My Passport/labels/lab/obj_left8.npy')
train_dat_path_lab4R= os.path.join('/media/timyang/My Passport/labels/lab/obj_right8.npy')
train_dat_path_office1L= os.path.join('/media/timyang/My Passport/labels/office/obj_left4.npy')
train_dat_path_office1R= os.path.join('/media/timyang/My Passport/labels/office/obj_right4.npy')
train_dat_path_office2L= os.path.join('/media/timyang/My Passport/labels/office/obj_left5.npy')
train_dat_path_office2R= os.path.join('/media/timyang/My Passport/labels/office/obj_right5.npy')
train_dat_path_office3L= os.path.join('/media/timyang/My Passport/labels/office/obj_left6.npy')
train_dat_path_office3R= os.path.join('/media/timyang/My Passport/labels/office/obj_right6.npy')
```


Results
---
一開始使用alexnet再做training發現結果不盡理想且速度太慢後來改進使用`VGG`來做training network
因為種類判別,
有特地將lable轉成`ont hot encodeing`
加上使用`cross entropy`來做loss funtion
optimizer使用`adamax`(原本使用`gradient decent`,model的名稱是`150x150(0.402).h5`,效果較差)
在訓練圖片時也有機進行翻轉處理

下圖中：關於training loss的部份有明顯下降,且accurancy有穩定進步,但可看出validation的部份卻沒有更好,可能已經overfitting


![](https://github.com/sun52525252/DL_HW_1/blob/master/preview/rslt.png)

最後測試 在`150x150`大小下,train `100` 個iterator有最高的accuracy(`150x150_50iter.h5(0.462)`),如下圖可看出在test data 中有6552為標記為free之照片數
最後可得accurancy為`0.506`

![](https://github.com/sun52525252/DL_HW_1/blob/master/preview/rslt3.png)
![](https://github.com/sun52525252/DL_HW_1/blob/master/preview/Figure_1.png)

![](https://github.com/sun52525252/DL_HW_1/blob/master/preview/Figure_1-2.png)

<br/>    
Data skewing problem:
發現training data 14992張中有7412張的label皆為free,造成training data 太過skew
所以在training時嘗試只讀取固定的free image,如model'200x200_free200(0.348).h5'
但最後發現當free取的越多accurancy越高,最後發現因該是因為testing data也將近一半都是free所造成

<br/>    
Why cannot get a high accurancy?
猜測無法提高accuracy原因如下：
    
1.label過於簡單,為標出物件所在位置,一張照片同時出現多個需要辨識物件的情況下會有衝突

2.label不一致,發現左右手在同一個畫面出現時label結果是不一樣的,雖拍攝角度不同但影像抓出的特徵可能十分相識,卻因label不同影響了學習
  如 train house 1 的第 723 張照片 上圖左手照片有拍到下圖右手拿餅乾label卻是free,但餅乾確實出現在畫面中
  
![](https://github.com/sun52525252/DL_HW_1/blob/master/preview/723l.png)

![](https://github.com/sun52525252/DL_HW_1/blob/master/preview/723r.png)

3.Data skewing problem:
  如上述,各樣data之數量因該相近,且test data何train data之同種之物體因該要更為相似

        
