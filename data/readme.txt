# Introduction

This is a dataset recorded by hand camera system.

The camera system consist of three wide-angle cameras, two mounted on the left and right wrists to
capture hands (referred to as HandCam) and one mounted on the head (referred to as HeadCam).

The dataset consists of 20 sets of video sequences (i.e., each set includes two HandCams and one
HeadCam synchronized videos) captured in three scenes: a small office, a mid-size lab, and a large home.)

We want to classify some kinds of hand states including free v.s. active (i.e., hands holding objects or not),
object categories, and hand gestures. At the same time, a synchronized video has two sequence need to be labeled,
the left hand states and right hand states.

For each classification task (i.e., free vs. active, object categories, or hand gesture), there are forty
sequences of data. We split the dataset into two parts, half for training, half for testing. The object instance
is totally separated into training and testing.


# Zip files

frames.zip contains all the frames sample from the original videos by 6fps.

labels.zip conatins the labels for all frames.
FA : free vs. active (only 0/1)
obj: object categories (24 classes, including free)
ges: hand gesture (13 gestures, including free)


# Details of obj. and ges.
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

Ges= {  'free':0,
        'press'1,
        'large-diameter':2,
        'lateral-tripod':3,
        'parallel-extension':4,
        'thumb-2-finger':5,
        'thumb-4-finger':6,
        'thumb-index-finger':7,
        'precision-disk':8,
        'lateral-pinch':9,
        'tripod':10,
        'medium-wrap':11,
        'light-tool':12}


