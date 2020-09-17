import tensorflow as tf
from src.utils import config


class TFRecordLoader:
    def __init__(self, batch_size, nb_epoch, nb_classes=20):
        super(TFRecordLoader, self).__init__()
        self.nb_epoch = nb_epoch
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.feature_description = {'path': tf.io.FixedLenFeature([], tf.string),
                                    's_label': tf.io.FixedLenFeature([], tf.string),
                                    'm_label': tf.io.FixedLenFeature([], tf.string),
                                    'l_label': tf.io.FixedLenFeature([], tf.string),
                                    's_boxes': tf.io.FixedLenFeature([], tf.string),
                                    'm_boxes': tf.io.FixedLenFeature([], tf.string),
                                    'l_boxes': tf.io.FixedLenFeature([], tf.string)}

    def parse_data(self, tf_record):
        features = tf.io.parse_single_example(tf_record, self.feature_description)

        image = tf.io.read_file(features['path'])
        image = tf.io.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image = tf.cast(image, tf.float32)
        image = image / 255.

        s_label = tf.io.decode_raw(features['s_label'], tf.float32)
        s_label = tf.reshape(s_label, (64, 64, 3, 5 + self.nb_classes))

        m_label = tf.io.decode_raw(features['m_label'], tf.float32)
        m_label = tf.reshape(m_label, (32, 32, 3, 5 + self.nb_classes))

        l_label = tf.io.decode_raw(features['l_label'], tf.float32)
        l_label = tf.reshape(l_label, (16, 16, 3, 5 + self.nb_classes))

        s_boxes = tf.io.decode_raw(features['s_boxes'], tf.float32)
        s_boxes = tf.reshape(s_boxes, (config.max_boxes, 4))

        m_boxes = tf.io.decode_raw(features['m_boxes'], tf.float32)
        m_boxes = tf.reshape(m_boxes, (config.max_boxes, 4))

        l_boxes = tf.io.decode_raw(features['l_boxes'], tf.float32)
        l_boxes = tf.reshape(l_boxes, (config.max_boxes, 4))

        return image, s_label, s_boxes, m_label, m_boxes, l_label, l_boxes

    def load_data(self, file_names):
        reader = tf.data.TFRecordDataset(file_names)
        # reader = reader.shuffle(len(file_names))
        reader = reader.map(self.parse_data, tf.data.experimental.AUTOTUNE)
        reader = reader.repeat(self.nb_epoch + 1)
        reader = reader.batch(self.batch_size)
        reader = reader.prefetch(tf.data.experimental.AUTOTUNE)
        return reader
