from __future__ import print_function, division

import cv2


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_image(path, image):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def load_annotations(txt_list):
    with open(txt_list, "r") as reader:
        lines = reader.readlines()
    return [line.strip() for line in lines if len(line.strip().split()[1:]) != 0]


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
