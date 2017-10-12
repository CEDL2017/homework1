import os
# Image fundemential information
img_height = 200
img_width = 200
img_channel = 3

# Mapping object (Defined)
# obj_name_2_index = {
#     'butterfly': 0,
#     'gate': 1,
#     'flower': 2,
#     'lenwen': 3,
#     'library': 4,
#     'law': 5
# }
# obj_name_2_index = {
#     'gate': 0,
#     'flower': 1,
#     'lenwen': 2,
#     'library': 3,
#     'law': 4
# }
# obj_index_2_color_tuple = {
#     0: (100, 0, 0),
#     1: (200, 0, 0),
#     2: (150, 0, 0),
#     3: (250, 0, 0),
#     4: (250, 0, 100),
#     # 5: (255, 0, 100)
# }
# obj_index_2_response_color_tuple = {
#     0: (0, 40, 0),
#     1: (0, 80, 0),
#     2: (0, 120, 0),
#     3: (0, 160, 0),
#     4: (0, 200, 0),
#     # 5: (0, 240, 0)
# }
obj_name_2_index = {
        'free':0,
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
        'lamp-switch':23
}
obj_index_2_color_tuple = {
    0: (100, 0, 0),
    1: (200, 0, 0),
    2: (150, 0, 0),
    3: (250, 0, 0),
    4: (250, 0, 100),
    5: (255, 0, 100),
    6: (255, 0, 100),
    7: (255, 0, 100),
    8: (255, 0, 100),
    9: (255, 0, 100),
    10: (255, 0, 100),
    11: (255, 0, 100),
    12: (255, 0, 100),
    13: (255, 0, 100),
    14: (255, 0, 100),
    15: (255, 0, 100),
    16: (255, 0, 100),
    17: (255, 0, 100),
    18: (255, 0, 100),
    20: (255, 0, 100),
    21: (255, 0, 100),
    22: (255, 0, 100),
    23: (255, 0, 100),
}
obj_index_2_response_color_tuple = {
    0: (0, 40, 0),
    1: (0, 80, 0),
    2: (0, 120, 0),
    3: (0, 160, 0),
    4: (0, 200, 0),
    5: (0, 240, 0),
    6: (0, 240, 0),
    7: (0, 240, 0),
    8: (0, 240, 0),
    9: (0, 240, 0),
    10: (0, 240, 0),
    11: (0, 240, 0),
    12: (0, 240, 0),
    13: (0, 240, 0),
    14: (0, 240, 0),
    15: (0, 240, 0),
    16: (0, 240, 0),
    17: (0, 240, 0),
    18: (0, 240, 0),
    20: (0, 240, 0),
    21: (0, 240, 0),
    22: (0, 240, 0),
    23: (0, 240, 0),
}
# ------------------------------------------------------------------
# The video capture name
# (If you want to life demo, set as 0 or other video device index)
# ------------------------------------------------------------------
video_name = '../video/3.mp4'

# The index of fram we want to break 
break_frame_index = 500

# Keras training epoch
general_epoch = 100

# Grid number (Defined)
# grid_height_num = 9
# grid_width_num = 16

grid_height_num = 1
grid_width_num = 1

# Model constants
model_path = '/home/viplab/Downloads/labels/'
unet_model_name = 'unet.h5'
scorenet_model_name = 'scorenet.h5' 

# ScoreNet training data path
scorenet_img_path = '/home/viplab/Downloads/frames/train/house/1/Lhand/'
scorenet_dat_path = '../dat/'
train_img_path = '/home/viplab/Downloads/frames/train/house/1/Lhand/'


# scorenet_img_path_house1L = os.path.join('/media/timyang/My Passport/frames/train/house/1/Lhand/')
# scorenet_img_path_house1R = os.path.join('/media/timyang/My Passport/frames/train/house/1/Rhand/')
# scorenet_img_path_house2L = os.path.join('/media/timyang/My Passport/frames/train/house/2/Lhand/')
# scorenet_img_path_house2R = os.path.join('/media/timyang/My Passport/frames/train/house/2/Rhand/')
# scorenet_img_path_house3L = os.path.join('/media/timyang/My Passport/frames/train/house/3/Lhand/')
# scorenet_img_path_house3R = os.path.join('/media/timyang/My Passport/frames/train/house/3/Rhand/')
# scorenet_img_path_lab1L= os.path.join('/media/timyang/My Passport/frames/train/lab/1/Lhand/')
# scorenet_img_path_lab1R= os.path.join('/media/timyang/My Passport/frames/train/lab/1/Rhand/')
# scorenet_img_path_lab2L= os.path.join('/media/timyang/My Passport/frames/train/lab/2/Lhand/')
# scorenet_img_path_lab2R= os.path.join('/media/timyang/My Passport/frames/train/lab/2/Rhand/')
# scorenet_img_path_lab3L= os.path.join('/media/timyang/My Passport/frames/train/lab/3/Lhand/')
# scorenet_img_path_lab3R= os.path.join('/media/timyang/My Passport/frames/train/lab/3/Rhand/')
# scorenet_img_path_lab4L= os.path.join('/media/timyang/My Passport/frames/train/lab/4/Lhand/')
# scorenet_img_path_lab4R= os.path.join('/media/timyang/My Passport/frames/train/lab/4/Rhand/')
# scorenet_img_path_office1L = os.path.join('/media/timyang/My Passport/frames/train/office/1/Lhand/')
# scorenet_img_path_office1R = os.path.join('/media/timyang/My Passport/frames/train/office/1/Rhand/')
# scorenet_img_path_office2L = os.path.join('/media/timyang/My Passport/frames/train/office/2/Lhand/')
# scorenet_img_path_office2R = os.path.join('/media/timyang/My Passport/frames/train/office/2/Rhand/')
# scorenet_img_path_office3L = os.path.join('/media/timyang/My Passport/frames/train/office/3/Lhand/')
# scorenet_img_path_office3R = os.path.join('/media/timyang/My Passport/frames/train/office/3/Rhand/')
scorenet_img_path_house1L = os.path.join('/home/viplab/Downloads/frames/train/house/1/Lhand/')
scorenet_img_path_house1R = os.path.join('/home/viplab/Downloads/frames/train/house/1/Rhand/')
scorenet_img_path_house2L = os.path.join('/home/viplab/Downloads/frames/train/house/2/Lhand/')
scorenet_img_path_house2R = os.path.join('/home/viplab/Downloads/frames/train/house/2/Rhand/')
scorenet_img_path_house3L = os.path.join('/home/viplab/Downloads/frames/train/house/3/Lhand/')
scorenet_img_path_house3R = os.path.join('/home/viplab/Downloads/frames/train/house/3/Rhand/')
scorenet_img_path_lab1L= os.path.join('/home/viplab/Downloads/frames/train/lab/1/Lhand/')
scorenet_img_path_lab1R= os.path.join('/home/viplab/Downloads/frames/train/lab/1/Rhand/')
scorenet_img_path_lab2L= os.path.join('/home/viplab/Downloads/frames/train/lab/2/Lhand/')
scorenet_img_path_lab2R= os.path.join('/home/viplab/Downloads/frames/train/lab/2/Rhand/')
scorenet_img_path_lab3L= os.path.join('/home/viplab/Downloads/frames/train/lab/3/Lhand/')
scorenet_img_path_lab3R= os.path.join('/home/viplab/Downloads/frames/train/lab/3/Rhand/')
scorenet_img_path_lab4L= os.path.join('/home/viplab/Downloads/frames/train/lab/4/Lhand/')
scorenet_img_path_lab4R= os.path.join('/home/viplab/Downloads/frames/train/lab/4/Rhand/')
scorenet_img_path_office1L = os.path.join('/home/viplab/Downloads/frames/train/office/1/Lhand/')
scorenet_img_path_office1R = os.path.join('/home/viplab/Downloads/frames/train/office/1/Rhand/')
scorenet_img_path_office2L = os.path.join('/home/viplab/Downloads/frames/train/office/2/Lhand/')
scorenet_img_path_office2R = os.path.join('/home/viplab/Downloads/frames/train/office/2/Rhand/')
scorenet_img_path_office3L = os.path.join('/home/viplab/Downloads/frames/train/office/3/Lhand/')
scorenet_img_path_office3R = os.path.join('/home/viplab/Downloads/frames/train/office/3/Rhand/')

train_img_path_house1L = os.path.join('/home/viplab/Downloads/frames/test/house/1/Lhand/')
train_img_path_house1R = os.path.join('/home/viplab/Downloads/frames/test/house/1/Rhand/')
train_img_path_house2L = os.path.join('/home/viplab/Downloads/frames/test/house/2/Lhand/')
train_img_path_house2R = os.path.join('/home/viplab/Downloads/frames/test/house/2/Rhand/')
train_img_path_house3L = os.path.join('/home/viplab/Downloads/frames/test/house/3/Lhand/')
train_img_path_house3R = os.path.join('/home/viplab/Downloads/frames/test/house/3/Rhand/')
train_img_path_lab1L= os.path.join('/home/viplab/Downloads/frames/test/lab/1/Lhand/')
train_img_path_lab1R= os.path.join('/home/viplab/Downloads/frames/test/lab/1/Rhand/')
train_img_path_lab2L= os.path.join('/home/viplab/Downloads/frames/test/lab/2/Lhand/')
train_img_path_lab2R= os.path.join('/home/viplab/Downloads/frames/test/lab/2/Rhand/')
train_img_path_lab3L= os.path.join('/home/viplab/Downloads/frames/test/lab/3/Lhand/')
train_img_path_lab3R= os.path.join('/home/viplab/Downloads/frames/test/lab/3/Rhand/')
train_img_path_lab4L= os.path.join('/home/viplab/Downloads/frames/test/lab/4/Lhand/')
train_img_path_lab4R= os.path.join('/home/viplab/Downloads/frames/test/lab/4/Rhand/')
train_img_path_office1L = os.path.join('/home/viplab/Downloads/frames/test/office/1/Lhand/')
train_img_path_office1R = os.path.join('/home/viplab/Downloads/frames/test/office/1/Rhand/')
train_img_path_office2L = os.path.join('/home/viplab/Downloads/frames/test/office/2/Lhand/')
train_img_path_office2R = os.path.join('/home/viplab/Downloads/frames/test/office/2/Rhand/')
train_img_path_office3L = os.path.join('/home/viplab/Downloads/frames/test/office/3/Lhand/')
train_img_path_office3R = os.path.join('/home/viplab/Downloads/frames/test/office/3/Rhand/')


# scorenet_dat_path_zip = os.path.join(data_env,'labels.zip')
scorenet_dat_path_house1L = os.path.join('/home/viplab/Downloads/labels/house/obj_left1.npy')
scorenet_dat_path_house1R = os.path.join('/home/viplab/Downloads/labels/house/obj_right1.npy')
scorenet_dat_path_house2L= os.path.join('/home/viplab/Downloads/labels/house/obj_left2.npy')
scorenet_dat_path_house2R= os.path.join('/home/viplab/Downloads/labels/house/obj_right2.npy')
scorenet_dat_path_house3L= os.path.join('/home/viplab/Downloads/labels/house/obj_left3.npy')
scorenet_dat_path_house3R= os.path.join('/home/viplab/Downloads/labels/house/obj_right3.npy')
scorenet_dat_path_lab1L= os.path.join('/home/viplab/Downloads/labels/lab/obj_left1.npy')
scorenet_dat_path_lab1R= os.path.join('/home/viplab/Downloads/labels/lab/obj_right1.npy')
scorenet_dat_path_lab2L= os.path.join('/home/viplab/Downloads/labels/lab/obj_left2.npy')
scorenet_dat_path_lab2R= os.path.join('/home/viplab/Downloads/labels/lab/obj_right2.npy')
scorenet_dat_path_lab3L= os.path.join('/home/viplab/Downloads/labels/lab/obj_left3.npy')
scorenet_dat_path_lab3R= os.path.join('/home/viplab/Downloads/labels/lab/obj_right3.npy')
scorenet_dat_path_lab4L= os.path.join('/home/viplab/Downloads/labels/lab/obj_left4.npy')
scorenet_dat_path_lab4R= os.path.join('/home/viplab/Downloads/labels/lab/obj_right4.npy')
scorenet_dat_path_office1L= os.path.join('/home/viplab/Downloads/labels/office/obj_left1.npy')
scorenet_dat_path_office1R= os.path.join('/home/viplab/Downloads/labels/office/obj_right1.npy')
scorenet_dat_path_office2L= os.path.join('/home/viplab/Downloads/labels/office/obj_left2.npy')
scorenet_dat_path_office2R= os.path.join('/home/viplab/Downloads/labels/office/obj_right2.npy')
scorenet_dat_path_office3L= os.path.join('/home/viplab/Downloads/labels/office/obj_left3.npy')
scorenet_dat_path_office3R= os.path.join('/home/viplab/Downloads/labels/office/obj_right3.npy')

train_dat_path_house1L = os.path.join('/home/viplab/Downloads/labels/house/obj_left4.npy')
train_dat_path_house1R = os.path.join('/home/viplab/Downloads/labels/house/obj_right4.npy')
train_dat_path_house2L= os.path.join('/home/viplab/Downloads/labels/house/obj_left5.npy')
train_dat_path_house2R= os.path.join('/home/viplab/Downloads/labels/house/obj_right5.npy')
train_dat_path_house3L= os.path.join('/home/viplab/Downloads/labels/house/obj_left6.npy')
train_dat_path_house3R= os.path.join('/home/viplab/Downloads/labels/house/obj_right6.npy')
train_dat_path_lab1L= os.path.join('/home/viplab/Downloads/labels/lab/obj_left5.npy')
train_dat_path_lab1R= os.path.join('/home/viplab/Downloads/labels/lab/obj_right5.npy')
train_dat_path_lab2L= os.path.join('/home/viplab/Downloads/labels/lab/obj_left6.npy')
train_dat_path_lab2R= os.path.join('/home/viplab/Downloads/labels/lab/obj_right6.npy')
train_dat_path_lab3L= os.path.join('/home/viplab/Downloads/labels/lab/obj_left7.npy')
train_dat_path_lab3R= os.path.join('/home/viplab/Downloads/labels/lab/obj_right7.npy')
train_dat_path_lab4L= os.path.join('/home/viplab/Downloads/labels/lab/obj_left8.npy')
train_dat_path_lab4R= os.path.join('/home/viplab/Downloads/labels/lab/obj_right8.npy')
train_dat_path_office1L= os.path.join('/home/viplab/Downloads/labels/office/obj_left4.npy')
train_dat_path_office1R= os.path.join('/home/viplab/Downloads/labels/office/obj_right4.npy')
train_dat_path_office2L= os.path.join('/home/viplab/Downloads/labels/office/obj_left5.npy')
train_dat_path_office2R= os.path.join('/home/viplab/Downloads/labels/office/obj_right5.npy')
train_dat_path_office3L= os.path.join('/home/viplab/Downloads/labels/office/obj_left6.npy')
train_dat_path_office3R= os.path.join('/home/viplab/Downloads/labels/office/obj_right6.npy')


# The number of neuron in ScoreNet
# (Auto-generated)
scorenet_fc_num = len(obj_name_2_index) * grid_height_num * grid_width_num
