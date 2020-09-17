import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convolution_block(inputs, filters, kernel_size, strides=1, activate=True, bn=True):
    if strides == 2:
        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        padding = 'valid'
    else:
        padding = 'same'

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=not bn,
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               bias_initializer=tf.constant_initializer(0.))(inputs)

    if bn:
        x = BatchNormalization()(x)
    if activate:
        x = tf.nn.leaky_relu(x, alpha=0.1)

    return x


def residual_block(inputs, filter_num1, filter_num2):
    x = convolution_block(inputs, filter_num1, 1)
    x = convolution_block(x, filter_num2, 3)

    return inputs + x


def transpose(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


def backbone(inputs):
    x = convolution_block(inputs, 32, 3)
    x = convolution_block(x, 64, 3, 2)

    for _ in range(1):
        x = residual_block(x, 32, 64)

    x = convolution_block(x, 128, 3, 2)

    for _ in range(2):
        x = residual_block(x, 64, 128)

    x = convolution_block(x, 256, 3, 2)

    for _ in range(8):
        x = residual_block(x, 128, 256)

    skip1 = x
    x = convolution_block(x, 512, 3, 2)

    for _ in range(8):
        x = residual_block(x, 256, 512)

    skip2 = x
    x = convolution_block(x, 1024, 3, 2)

    for _ in range(4):
        x = residual_block(x, 512, 1024)

    return skip1, skip2, x
