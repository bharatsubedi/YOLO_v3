import os
import sys
from os.path import join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from src.utils import config

tf.random.set_seed(config.seed)

from src.nets import nn
from src.utils import util, data_loader

tf_path = join(config.dataset, 'record')
tf_paths = [join(tf_path, name) for name in os.listdir(tf_path) if name.endswith('.tfrecord')]

np.random.shuffle(tf_paths)

strategy = tf.distribute.MirroredStrategy()

nb_gpu = strategy.num_replicas_in_sync
global_batch = nb_gpu * config.batch_size
nb_classes = len(util.read_class_names(config.classes))

dataset = data_loader.TFRecordLoader(global_batch, config.nb_epoch, nb_classes).load_data(tf_paths)
dataset = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(0.0001)
    input_tensor = tf.keras.layers.Input([config.image_size, config.image_size, 3])
    outputs = nn.build_model(input_tensor, nb_classes)
    output_tensors = []
    for i, output in enumerate(outputs):
        pred_tensor = nn.decode(output, i)
        output_tensors.append(output)
        output_tensors.append(pred_tensor)
    model = tf.keras.Model(input_tensor, output_tensors)

print(f'[INFO] {len(tf_paths)} train data')

with strategy.scope():
    loss_object = nn.compute_loss


    def compute_loss(_target, _logits):
        _iou_loss = _conf_loss = _prob_loss = 0

        for ind in range(3):
            loss_items = loss_object(_logits[ind * 2 + 1], _logits[ind * 2], _target[ind][0], _target[ind][1], ind)
            _iou_loss += loss_items[0]
            _conf_loss += loss_items[1]
            _prob_loss += loss_items[2]

        _total_loss = _iou_loss + _conf_loss + _prob_loss

        _iou_loss = tf.reduce_sum(_iou_loss) * 1. / global_batch
        _conf_loss = tf.reduce_sum(_conf_loss) * 1. / global_batch
        _prob_loss = tf.reduce_sum(_prob_loss) * 1. / global_batch

        _total_loss = _iou_loss + _conf_loss + _prob_loss

        return _iou_loss, _conf_loss, _prob_loss, _total_loss

with strategy.scope():
    def train_step(_image, _target):
        with tf.GradientTape() as tape:
            _logits = model(_image, training=True)
            _iou_loss, _conf_loss, _prob_loss, _total_loss = compute_loss(_target, _logits)

            gradients = tape.gradient(_total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return _iou_loss, _conf_loss, _prob_loss, _total_loss

with strategy.scope():
    @tf.function
    def distribute_train_step(_image, _target):
        _iou_loss, _conf_loss, _prob_loss, _total_loss = strategy.run(train_step, args=(_image, _target))
        _total_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, _total_loss, axis=None)
        _iou_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, _iou_loss, axis=None)
        _conf_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, _conf_loss, axis=None)
        _prob_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, _prob_loss, axis=None)
        return _iou_loss, _conf_loss, _prob_loss, _total_loss

if __name__ == '__main__':
    nb_steps = len(tf_paths) // global_batch
    print(f"--- Training with {nb_steps} Steps ---")
    for step, inputs in enumerate(dataset):
        step += 1
        image, s_label, s_boxes, m_label, m_boxes, l_label, l_boxes = inputs
        target = ((s_label, s_boxes), (m_label, m_boxes), (l_label, l_boxes))
        iou_loss, conf_loss, prob_loss, total_loss = distribute_train_step(image, target)
        print("%4d  %4.2f  %4.2f  %4.2f  %4.2f" % (step,
                                                   iou_loss.numpy(),
                                                   conf_loss.numpy(),
                                                   prob_loss.numpy(),
                                                   total_loss.numpy()))
        if step % (10 * nb_steps) == 0:
            model.save_weights(join("weights", f"model{step // (10 * nb_steps)}.h5"))
        if step // nb_steps == config.nb_epoch:
            sys.exit("--- Stop Training ---")
