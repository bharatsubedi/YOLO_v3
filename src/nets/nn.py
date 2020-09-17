import numpy as np
import tensorflow as tf

from src.nets import helper
from src.utils import config


def build_model(inputs, nb_classes):
    x1, x2, x3 = helper.backbone(inputs)

    x = helper.convolution_block(x3, 512, 1)
    x = helper.convolution_block(x, 1024, 3)
    x = helper.convolution_block(x, 512, 1)
    x = helper.convolution_block(x, 1024, 3)
    x = helper.convolution_block(x, 512, 1)

    large = helper.convolution_block(x, 1024, 3)
    large = helper.convolution_block(large, 3 * (nb_classes + 5), 1, activate=False, bn=False)

    x = helper.convolution_block(x, 256, 1)
    x = helper.transpose(x)

    x = tf.concat([x, x2], axis=-1)

    x = helper.convolution_block(x, 256, 1)
    x = helper.convolution_block(x, 512, 3)
    x = helper.convolution_block(x, 256, 1)
    x = helper.convolution_block(x, 512, 3)
    x = helper.convolution_block(x, 256, 1)

    medium = helper.convolution_block(x, 512, 3)
    medium = helper.convolution_block(medium, 3 * (nb_classes + 5), 1, activate=False, bn=False)

    x = helper.convolution_block(x, 128, 1)
    x = helper.transpose(x)

    x = tf.concat([x, x1], axis=-1)

    x = helper.convolution_block(x, 128, 1)
    x = helper.convolution_block(x, 256, 3)
    x = helper.convolution_block(x, 128, 1)
    x = helper.convolution_block(x, 256, 3)
    x = helper.convolution_block(x, 128, 1)

    small = helper.convolution_block(x, 256, 3)
    small = helper.convolution_block(small, 3 * (nb_classes + 5), 1, activate=False, bn=False)

    return [small, medium, large]


def decode(output, i=0):
    shape = tf.shape(output)
    batch_size = shape[0]
    output_size = shape[1]

    output = tf.reshape(output, (batch_size, output_size, output_size, 3, 5 + 20))

    raw_xy = output[:, :, :, :, 0:2]
    raw_wh = output[:, :, :, :, 2:4]
    raw_conf = output[:, :, :, :, 4:5]
    raw_prob = output[:, :, :, :, 5:]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(raw_xy) + xy_grid) * config.strides[i]
    pred_wh = (tf.exp(raw_wh) * config.anchors[i]) * config.strides[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(raw_conf)
    pred_prob = tf.sigmoid(raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_g_iou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    return iou - 1.0 * (enclose_area - union_area) / enclose_area


def compute_loss(pred, output, label, boxes, i=0):
    shape = tf.shape(output)
    batch_size = shape[0]
    output_size = shape[1]
    input_size = config.strides[i] * output_size
    output = tf.reshape(output, (batch_size, output_size, output_size, 3, 5 + 20))

    raw_conf = output[:, :, :, :, 4:5]
    raw_prob = output[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_g_iou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], boxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < 0.5, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(respond_bbox, raw_conf)
                              +
                              respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(respond_bbox, raw_conf))

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(label_prob, raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
