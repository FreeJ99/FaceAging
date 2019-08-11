import tensorflow as tf


class FCGenerator:
    def __init__(self, img_size, channels):
        """
        Network which takes a batch of random vectors and creates images out of them with.

        :param img_size: width and height of the image
        :param channels: number of channels
        """
        self.img_size = img_size
        self.channels = channels

    def __call__(self, z):
        """
        Method which performs the computation.

        :param z: tensor of the shape [batch_size, z_size] representing batch_size random vectors from the
        prior distribution
        :return: image of the shape [batch_size, img_size, img_size, channels]
        """
        with tf.variable_scope("Generator"):
            z = tf.layers.dense(z, 512, activation=tf.nn.relu)
            z = tf.layers.dense(z, 512, activation=tf.nn.relu)
            z = tf.layers.dense(z, self.img_size * self.img_size * self.channels, activation=tf.nn.sigmoid)
            image = tf.reshape(z, [-1, self.img_size, self.img_size, self.channels])
            return image


class ConvGenerator:
    def __init__(self, img_size, channels):
        self.img_size = img_size
        self.channels = channels

    def __call__(self, z):
        with tf.variable_scope("Generator"):
            #cnn
            z = conv_layer(inputs=z, filters=32, kernel_size=[7, 7], strides=2, with_activation=True, name="conv1")
            z = pool_layer(inputs=z, pool_size=[2, 2], strides=2, name="pool1")
            z = fc_layer(inputs=z, units=1024, with_activation=True, name="fc1")
            z = fc_layer(inputs=z, units=100, with_activation=False, name="fc2")
            #generator
            act = tf.nn.relu
            res_met = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            pad2 = [[0, 0], [2, 2], [2, 2], [0, 0]]

            kwargs = {"strides": (1, 1), "padding": "valid"}

            z = tf.layers.dense(z, 32768, activation=act)
            z = tf.reshape(z, [-1, 4, 4, 2048])

            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=1024, kernel_size=(5, 5), **kwargs, activation=act)
            z = tf.image.resize_images(z, (16, 16), method=res_met)
            #
            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=512, kernel_size=(5, 5), **kwargs, activation=act)
            z = tf.image.resize_images(z, (32, 32), method=res_met)

            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=256, kernel_size=(5, 5), **kwargs, activation=act)
            z = tf.image.resize_images(z, (self.img_size, self.img_size), method=res_met)

            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=3, activation=tf.nn.sigmoid, kernel_size=(5, 5), **kwargs)
            return z

def conv_layer(inputs, filters, kernel_size, strides, with_activation, name):
    activation = tf.nn.relu if with_activation else None

    return tf.layers.conv2d(

        inputs=inputs,

        filters=filters,

        kernel_size=kernel_size,

        strides=strides,

        padding="SAME",

        activation=activation,

        name=name)
def pool_layer(inputs, pool_size, strides, name):
    return tf.layers.max_pooling2d(

        inputs=inputs,

        pool_size=pool_size,

        strides=strides,

        padding="SAME",

        name=name)

def fc_layer(inputs, units, with_activation, name):
    inputs_flat = inputs if len(inputs.shape) < 2 else tf.reshape(inputs, [-1, np.prod(inputs.shape[1:])])

    activation = tf.nn.relu if with_activation else None

    return tf.layers.dense(inputs=inputs_flat, units=units, activation=activation, name=name)




class DCGANGenerator:
    def __init__(self, img_size, channels):
        self.channels = channels

    def __call__(self, z):
        """

        :param z:
        :return: returns tensor of shape [batch_size, 64, 64, channels]
        """
        with tf.variable_scope("Generator"):
            act = tf.nn.relu

            z = tf.layers.dense(z, 32768, activation=act)
            z = tf.reshape(z, [-1, 4, 4, 2048])

            kwargs = {"kernel_size": (5, 5), "strides": (2, 2), "padding": "same"}

            z = tf.layers.conv2d_transpose(z, filters=512, activation=act, **kwargs)
            z = tf.layers.conv2d_transpose(z, filters=256, activation=act, **kwargs)
            z = tf.layers.conv2d_transpose(z, filters=128, activation=act, **kwargs)
            z = tf.layers.conv2d_transpose(z, filters=self.channels, activation=tf.nn.sigmoid, **kwargs)
            return z
