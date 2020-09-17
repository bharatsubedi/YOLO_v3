import multiprocessing
import os
from multiprocessing import Process
from os.path import join

import cv2
import numpy as np
import tensorflow as tf
import tqdm
from lxml import etree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.utils import config, util


def make_names():
    labels_dict = {}

    annotation_list = os.listdir(os.path.join(config.dataset, config.label_path))

    for annotation_file in annotation_list:
        p = os.path.join(os.path.join(config.dataset, config.label_path), annotation_file)
        root = etree.parse(p).getroot()
        names = root.xpath('//object/name')

        for n in names:
            labels_dict[n.text] = 0

    labels = list(labels_dict.keys())
    labels.sort()

    with open(config.classes, 'w') as writer:
        for label in labels:
            writer.writelines(label + '\n')

    print("done !!!")


def convert_annotation(list_txt, output_path, image_dir, anno_dir, class_names):
    image_ext = '.jpg'
    anno_ext = '.xml'

    with open(list_txt, 'r') as f, open(output_path, 'w') as wf:
        while True:
            line = f.readline().strip()
            if line is None or not line:
                break
            im_p = os.path.join(image_dir, line + image_ext)
            an_p = os.path.join(anno_dir, line + anno_ext)

            # Get annotation.
            root = etree.parse(an_p).getroot()
            b_boxes = root.xpath('//object/bndbox')
            names = root.xpath('//object/name')

            box_annotations = []
            for b, n in zip(b_boxes, names):
                name = n.text
                class_idx = class_names.index(name)

                xmin = b.find('xmin').text
                ymin = b.find('ymin').text
                xmax = b.find('xmax').text
                ymax = b.find('ymax').text
                box_annotations.append(','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_idx)]))

            annotation = os.path.abspath(im_p) + ' ' + ' '.join(box_annotations) + '\n'

            wf.write(annotation)


def convert():
    image_dir = os.path.join(config.dataset, config.image_path)
    annotation_dir = os.path.join(config.dataset, config.label_path)
    train_list_txt = os.path.join(config.dataset, 'ImageSets', 'Main', 'train.txt')
    val_list_txt = os.path.join(config.dataset, 'ImageSets', 'Main', 'val.txt')
    train_output = os.path.join(config.dataset, 'voc2012_train.txt')
    val_output = os.path.join(config.dataset, 'voc2012_val.txt')

    class_names = [c.strip() for c in open(config.classes).readlines()]

    convert_annotation(train_list_txt, train_output, image_dir, annotation_dir, class_names)

    convert_annotation(val_list_txt, val_output, image_dir, annotation_dir, class_names)


def byte_feature(value):
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bbox_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5, ], axis=-1, )
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5, ], axis=-1, )

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / union_area


def preprocess(bboxes):
    num_classes = len(util.read_class_names(config.classes))
    train_output_sizes = [64, 32, 16]
    label = [np.zeros((train_output_sizes[i], train_output_sizes[i], 3, 5 + num_classes)) for i in range(3)]
    bboxes_xywh = [np.zeros((config.max_boxes, 4)) for _ in range(3)]
    bbox_count = np.zeros((3,))

    for bbox in bboxes:
        bbox_coordinate = bbox[:4]
        bbox_class_ind = bbox[4]

        one_hot = np.zeros(num_classes, dtype=np.float)
        one_hot[bbox_class_ind] = 1.0
        uniform_distribution = np.full(num_classes, 1.0 / num_classes)
        delta = 0.01
        smooth_one_hot = one_hot * (1 - delta) + delta * uniform_distribution

        bbox_xywh = np.concatenate(
            [(bbox_coordinate[2:] + bbox_coordinate[:2]) * 0.5, bbox_coordinate[2:] - bbox_coordinate[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / np.array(config.strides)[:, np.newaxis]

        iou = []
        exist_positive = False
        for i in range(3):
            anchors_xywh = np.zeros((3, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = config.anchors[i]

            iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > 0.3

            if np.any(iou_mask):
                x_ind, y_ind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                label[i][y_ind, x_ind, iou_mask, :] = 0
                label[i][y_ind, x_ind, iou_mask, 0:4] = bbox_xywh
                label[i][y_ind, x_ind, iou_mask, 4:5] = 1.0
                label[i][y_ind, x_ind, iou_mask, 5:] = smooth_one_hot

                bbox_ind = int(bbox_count[i] % config.max_boxes)
                bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                bbox_count[i] += 1

                exist_positive = True

        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / 3)
            best_anchor = int(best_anchor_ind % 3)
            x_ind, y_ind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

            label[best_detect][y_ind, x_ind, best_anchor, :] = 0
            label[best_detect][y_ind, x_ind, best_anchor, 0:4] = bbox_xywh
            label[best_detect][y_ind, x_ind, best_anchor, 4:5] = 1.0
            label[best_detect][y_ind, x_ind, best_anchor, 5:] = smooth_one_hot

            bbox_ind = int(bbox_count[best_detect] % config.max_boxes)
            bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
            bbox_count[best_detect] += 1
    l_s_box, l_m_box, l_l_box = label
    s_boxes, m_boxes, l_boxes = bboxes_xywh
    return l_s_box, l_m_box, l_l_box, s_boxes, m_boxes, l_boxes


def resize(image, gt_boxes=None):
    ih, iw = config.image_size, config.image_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.zeros(shape=[ih, iw, 3], dtype=np.uint8)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized.copy()

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes


def parse_annotation(annotation):
    line = annotation.split()
    image_path = line[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " % image_path)
    image = cv2.imread(image_path)

    bboxes = np.array([list(map(int, box.split(","))) for box in line[1:]])

    image, bboxes = resize(np.copy(image), bboxes)
    return image, bboxes, image_path


def build_example(annotation):
    image, bboxes, image_path = parse_annotation(annotation)
    s_label, m_label, l_label, s_boxes, m_boxes, l_boxes = preprocess(bboxes)

    path = os.path.join(config.dataset, 'record', os.path.basename(image_path))

    util.write_image(path, image)

    s_label = s_label.astype('float32')
    m_label = m_label.astype('float32')
    l_label = l_label.astype('float32')
    s_boxes = s_boxes.astype('float32')
    m_boxes = m_boxes.astype('float32')
    l_boxes = l_boxes.astype('float32')

    s_label = s_label.tobytes()
    m_label = m_label.tobytes()
    l_label = l_label.tobytes()

    s_boxes = s_boxes.tobytes()
    m_boxes = m_boxes.tobytes()
    l_boxes = l_boxes.tobytes()

    features = tf.train.Features(feature={'path': byte_feature(path.encode('utf-8')),
                                          's_label': byte_feature(s_label),
                                          'm_label': byte_feature(m_label),
                                          'l_label': byte_feature(l_label),
                                          's_boxes': byte_feature(s_boxes),
                                          'm_boxes': byte_feature(m_boxes),
                                          'l_boxes': byte_feature(l_boxes)})

    return tf.train.Example(features=features)


def write_tf_record(_queue, _sentinel):
    while True:
        annotation = _queue.get()

        if annotation == _sentinel:
            break
        tf_example = build_example(annotation)
        name = os.path.basename(annotation.split(' ')[0])
        annotation = join(config.dataset, 'record')
        if not os.path.exists(annotation):
            os.makedirs(annotation)
        with tf.io.TFRecordWriter(join(annotation, name[:-4] + ".tfrecord")) as writer:
            writer.write(tf_example.SerializeToString())


def main():
    if not os.path.exists(config.classes):
        make_names()
    if not os.path.exists(os.path.join(config.dataset, 'voc2012_train.txt')):
        convert()

    sentinel = ("", [])
    nb_process = multiprocessing.cpu_count()
    _queue = multiprocessing.Manager().Queue()
    annotations = util.load_annotations(os.path.join(config.dataset, 'voc2012_train.txt'))
    for annotation in tqdm.tqdm(annotations):
        _queue.put(annotation)
    for _ in range(nb_process):
        _queue.put(sentinel)
    print('--- generating tfrecord')
    process_pool = []
    for i in range(nb_process):
        process = Process(target=write_tf_record, args=(_queue, sentinel))
        process_pool.append(process)
        process.start()
    for process in process_pool:
        process.join()


if __name__ == '__main__':
    main()
