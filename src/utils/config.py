import numpy as np

seed = 12345
nb_epoch = 50
batch_size = 1
max_boxes = 150
image_size = 512

dataset = '../VOC2012'
image_path = 'JPEGImages'
label_path = 'Annotations'
classes = '../VOC2012/voc2012.names'
strides = [8, 16, 32]
anchors = np.array([[[12, 16], [19, 36], [40, 28]],
                    [[36, 75], [76, 55], [72, 146]],
                    [[142, 110], [192, 243], [459, 401]]], np.float32)
