"""
This module provides implementations for two version of PSPNet.

The 1st variant, PSPNet34, we use interally at KeyMe for our own semantic segmentation task.
The 2nd variant, PSPNet50, is a copy of the original caffe prototxt architecture provided by
https://github.com/hszhao/PSPNet and comes with pretrained ADE20K weights courtesy of
https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow.

Paper can be found here: https://arxiv.org/abs/1612.01105
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


def pyramid_pooling_module(input_layer, filters, pool_sizes, strides_list,
                                 level_indices, is_training, name_prefix):
    """
    Pyramid pooling module.

    This module will create a level for every triplet in
    `zip(pool_sizes, strides_list, level_indices)`.

    `level_indices` is only used in the naming of layers in each level and is
    necessary to keep this module arbitrary but still allow for original
    paper's caffe converted weights to be loaded.

    This module assumes data_format is `channels_last`.
    """

    def build_level(input_layer, pool_size, strides, filters, is_training,
                    name_prefix, level_index):
        """
        Helper function to build an individual level of the pyramid
        scene parsing module.

        `name_prefix` and `level_index` params are used in inter/intra-level
        naming and are only needed to match original prototxt layer names
        """

        level_pool = tf.layers.average_pooling2d(input_layer,
                                                  pool_size=pool_size,
                                                  strides=strides,
                                                  name="{}_pool{}".format(
                                                    name_prefix, level_index))

        level_conv = tf.layers.conv2d(level_pool,
                                       filters=filters,
                                       kernel_size=1,
                                       strides=1,
                                       use_bias=False,
                                       name="{}_pool{}_conv".format(
                                        name_prefix, level_index))

        level_bn = tf.layers.batch_normalization(
            level_conv, training=is_training,
            name="{}_pool{}_conv_bn".format(name_prefix, level_index))

        level_relu = tf.nn.relu(level_bn)

        level_interp = tf.image.resize_bilinear(
            level_relu, input_shape[1:-1], align_corners=True,
            name="{}_pool{}_interp".format(name_prefix, level_index))

        return level_interp


    assert len(pool_sizes) == len(strides_list) == len(level_indices)

    input_shape = tf.shape(input_layer)

    with tf.variable_scope("pyramid_pooling_module"):

        concat_layers = [input_layer]
        for pool_size, strides, level_index in zip(
            pool_sizes, strides_list, level_indices):

            level_interp = build_level(input_layer,
                                       pool_size=pool_size,
                                       strides=strides,
                                       filters=filters,
                                       is_training=is_training,
                                       name_prefix=name_prefix,
                                       level_index=level_index)

            concat_layers.append(level_interp)

        concat = tf.concat(concat_layers, axis=3)

    return concat


class PSPNet(ABC):
    """Implement abstract base for PSPNet architecture."""
    def __init__(self, input_shape, network_storage_dir, num_classes, learning_rate,
                 is_grayscale=False):
        """Construct the PSPNet object."""
        self.input_shape = input_shape
        self.network_weights_loc = network_storage_dir
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_channels = 1 if is_grayscale else 3

        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():

            # Set up the placeholder tensors to use as input/output to network
            keep_probability = tf.placeholder(tf.float32, name="keep_probability")
            image = tf.placeholder(
                tf.float32, shape=[None, input_shape[0], input_shape[1], self.num_channels],
                name="input_image")
            annotation = tf.placeholder(
                tf.int32, shape=[None, input_shape[0], input_shape[1], 1],
                name="annotation")
            is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

            self.placeholders = {"image": image,
                                 "keep_probability": keep_probability,
                                 "annotation": annotation,
                                 "is_training": is_training}

            # Build the computational graph
            self.logits, self.probabilities, self.predict = self.build_computational_graph()

            # Set up the saver and try to restore any model previously being trained
            self.saver = tf.train.Saver()

    @abstractmethod
    def build_computational_graph():
        """Contruct the computational graph for PSPNet."""
        pass

    def load_network_weights(self, path_to_network):
        """Load the network weights contained in `path_to_network` into memory."""
        ckpt = tf.train.get_checkpoint_state(path_to_network)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.weights_loaded = True
        else:
            self.weights_loaded = False

    def add_train_ops(self, loss_val, var_list, global_step):
        """Add training operations to computational graph."""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads = optimizer.compute_gradients(loss_val, var_list=var_list)
            return optimizer.apply_gradients(grads, global_step=global_step)


class PSPNet34(PSPNet):
    """Construct a PSPNet with backbone based on dilated ResNet34."""

    def build_computational_graph(self):
        """Build the PSPNet with ResNet34 backbone. and 4 level PSP module."""

        def residual_block(inputs, filters, kernel_size, scope, is_training, strides=1,
                           dilation_rate=1, data_format="channels_last"):
            """
            Construct a (possibly dilated) residual block.

            This block is based on the 3x3, 64 -> 3x3, 64 block architecture.
            """

            shortcut = inputs

            # Forbid striding and dilating in same block
            if strides not in [1, (1, 1)]:
                assert dilation_rate == 1
            if dilation_rate != 1:
                assert strides in [1, (1, 1)]

            with tf.variable_scope(scope):

                x = tf.layers.conv2d(
                    inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                    use_bias=False, dilation_rate=dilation_rate, padding="same")
                x = tf.nn.relu(x)
                x = tf.layers.batch_normalization(x, training=is_training)

                x = tf.layers.conv2d(
                    x, filters=filters, kernel_size=kernel_size, strides=1,
                    use_bias=False, dilation_rate=dilation_rate, padding="same")
                x = tf.layers.batch_normalization(x, training=is_training)

                if data_format == "channels_last":
                    _, h, w, d = inputs.get_shape().as_list()
                    _, hh, ww, dd = x.get_shape().as_list()
                else:
                    _, d, h, w = inputs.get_shape().as_list()
                    _, dd, hh, ww = x.get_shape().as_list()

                # If input and output shapes don't match, project input to output shape
                if h != hh or d != dd:
                    conv_proj = tf.layers.conv2d(
                        inputs, filters, kernel_size=1, strides=strides, use_bias=False)
                    conv_proj_bn = tf.layers.batch_normalization(conv_proj, training=is_training)
                    out = x + conv_proj_bn
                else:
                    out = x + inputs

                return tf.nn.relu(out)

        def build_dilated_residual_network(input_layer):
            """Construct 34-layer dilated residual network."""
            is_training = self.placeholders["is_training"]

            conv1 = tf.layers.conv2d(input_layer, filters=64, kernel_size=7, strides=2, padding="same", name="conv1")

            pool1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=2, padding='same', name="pool1")

            conv2_1 = residual_block(pool1, filters=64, kernel_size=3, is_training=is_training, strides=1, scope="conv2_1")
            conv2_2 = residual_block(conv2_1, filters=64, kernel_size=3, is_training=is_training, strides=1, scope="conv2_2")
            conv2_3 = residual_block(conv2_2, filters=64, kernel_size=3, is_training=is_training, strides=1, scope="conv2_3")

            conv3_1 = residual_block(conv2_3, filters=128, kernel_size=3, is_training=is_training, strides=2, scope="conv3_1")
            conv3_2 = residual_block(conv3_1, filters=128, kernel_size=3, is_training=is_training, strides=1, scope="conv3_2")
            conv3_3 = residual_block(conv3_2, filters=128, kernel_size=3, is_training=is_training, strides=1, scope="conv3_3")
            conv3_4 = residual_block(conv3_3, filters=128, kernel_size=3, is_training=is_training, strides=1, scope="conv3_4")

            # Replace stride of 2 with dilation in G4 and G5 as in DRN paper
            conv4_1 = residual_block(conv3_4, filters=256, kernel_size=3, is_training=is_training, dilation_rate=2, scope="conv4_1")
            conv4_2 = residual_block(conv4_1, filters=256, kernel_size=3, is_training=is_training, dilation_rate=2, scope="conv4_2")
            conv4_3 = residual_block(conv4_2, filters=256, kernel_size=3, is_training=is_training, dilation_rate=2, scope="conv4_3")
            conv4_4 = residual_block(conv4_3, filters=256, kernel_size=3, is_training=is_training, dilation_rate=2, scope="conv4_4")
            conv4_5 = residual_block(conv4_4, filters=256, kernel_size=3, is_training=is_training, dilation_rate=2, scope="conv4_5")
            conv4_6 = residual_block(conv4_5, filters=256, kernel_size=3, is_training=is_training, dilation_rate=2, scope="conv4_6")

            conv5_1 = residual_block(conv4_6, filters=512, kernel_size=3, is_training=is_training, dilation_rate=4, scope="conv5_1")
            conv5_2 = residual_block(conv5_1, filters=512, kernel_size=3, is_training=is_training, dilation_rate=4, scope="conv5_2")
            conv5_3 = residual_block(conv5_2, filters=512, kernel_size=3, is_training=is_training, dilation_rate=4, scope="conv5_3")

            return conv5_3

        input_shape = tf.shape(self.placeholders["image"])
        processed_image = self.placeholders["image"] / 255.0

        with tf.variable_scope("PSPNet"):

            # Use 34-layer dilated residual network as base
            drn = build_dilated_residual_network(processed_image)
            # Put pyramid pooling module over top of base drn
            psp = pyramid_pooling_module(drn,
                                         filters=512,
                                         pool_sizes=[1, 2, 3, 6],
                                         strides_list=[1, 2, 3, 6],
                                         level_indices=[1, 2, 3, 6],
                                         is_training=self.placeholders["is_training"],
                                         name_prefix="conv5_3")

            # Add final conv to get raw prediction
            conv = tf.layers.conv2d(psp,
                                    filters=self.num_classes,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    name="conv6")

            # Resize raw prediction to input size and add a tensor for prediction
            logits = tf.image.resize_bilinear(
                conv, size=self.input_shape, align_corners=True, name="logits")
            probs = tf.nn.softmax(logits)
            output = tf.argmax(logits, axis=3, name="argmax_up")
            output = tf.expand_dims(output, axis=3, name="predict")

            return logits, probs, output


class PSPNet50(PSPNet):
    """
    Construct a PSPNet with backbone based on dilated ResNet50 and which can
    load pretrained PSPNet ADE20K weights.
    """

    def build_computational_graph(self):
        """Build the PSPNet with ResNet50 backbone. and 4 level PSP module."""

        def bottleneck_module(inputs, lvl, pad, is_training, filters, strides,
                              data_format='channels_last', bottleneck_factor=4):
            """
            Implement the bottleneck module proposed in ResNet.
            1x1 conv -> 3x3 conv -> 1x1 conv
            """

            def zero_padding(input, paddings):
                """Zero padding layer (assumes NHWC format)."""
                pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
                return tf.pad(input, paddings=pad_mat)

            inputs = tf.nn.relu(inputs)

            # 1x1 reduce component
            x = tf.layers.conv2d(inputs, filters=filters // bottleneck_factor,
                                 kernel_size=1, strides=strides, data_format=data_format,
                                 use_bias=False, name="conv{}_1x1_reduce".format(lvl))
            x = tf.layers.batch_normalization(
                x, training=is_training, name="conv{}_1x1_reduce_bn".format(lvl))
            x = tf.nn.relu(x)

            # 3x3 component
            x = zero_padding(x, pad)
            x = tf.layers.conv2d(x, filters=filters // bottleneck_factor,
                                 kernel_size=3, strides=1, dilation_rate=pad,
                                 data_format=data_format, use_bias=False,
                                 name="conv{}_3x3".format(lvl))
            x = tf.layers.batch_normalization(
                x, training=is_training, name="conv{}_3x3_bn".format(lvl))
            x = tf.nn.relu(x)

            # 1x1 increase component
            x = tf.layers.conv2d(x, filters=filters, kernel_size=1, strides=1,
                                 data_format=data_format, use_bias=False,
                                 name="conv{}_1x1_increase".format(lvl))
            x = tf.layers.batch_normalization(
                x, training=is_training, name="conv{}_1x1_increase_bn".format(lvl))

            # 1x1 project (if needed)
            if data_format == "channels_last":
                _, h, w, d = inputs.get_shape().as_list()
                _, hh, ww, dd = x.get_shape().as_list()
            else:
                _, d, h, w = inputs.get_shape().as_list()
                _, dd, hh, ww = x.get_shape().as_list()

            if h != hh or d != dd:
                conv_proj = tf.layers.conv2d(inputs, filters, kernel_size=1,
                                             strides=strides, use_bias=False,
                                             name="conv{}_1x1_proj".format(lvl))
                conv_proj_bn = tf.layers.batch_normalization(
                    conv_proj, training=is_training, name="conv{}_1x1_proj_bn".format(lvl))
                out = x + conv_proj_bn
            else:
                out = x + inputs

            return out

        def build_dilated_residual_network(input_layer):
            """Construct a 34-layer variant dilated residual network."""
            is_training = self.placeholders["is_training"]

            conv1_1 = tf.layers.conv2d(
                input_layer, filters=64, kernel_size=3, strides=2,
                padding="same", use_bias=False, name="conv1_1_3x3_s2")
            conv1_1_bn = tf.layers.batch_normalization(
                conv1_1, training=is_training, name="conv1_1_3x3_s2_bn")
            conv1_1_relu = tf.nn.relu(conv1_1_bn)

            conv1_2 = tf.layers.conv2d(
                conv1_1_relu, filters=64, kernel_size=3, strides=1,
                padding="same", use_bias=False, name="conv1_2_3x3")
            conv1_2_bn = tf.layers.batch_normalization(
                conv1_2, training=is_training, name="conv1_2_3x3_bn")
            conv1_2_relu = tf.nn.relu(conv1_2_bn)

            conv1_3 = tf.layers.conv2d(
                conv1_2_relu, filters=128, kernel_size=3, strides=1,
                padding="same", use_bias=False, name="conv1_3_3x3")
            conv1_3_bn = tf.layers.batch_normalization(
                conv1_3, training=is_training, name="conv1_3_3x3_bn")
            conv1_3_relu = tf.nn.relu(conv1_3_bn)

            pool1 = tf.layers.max_pooling2d(
                conv1_3_relu, pool_size=3, strides=2, padding='same', name="pool1")

            conv2_1_block = bottleneck_module(pool1, lvl="2_1", pad=1, is_training=is_training, filters=256, strides=1)
            conv2_2_block = bottleneck_module(conv2_1_block, lvl="2_2", pad=1, is_training=is_training, filters=256, strides=1)
            conv2_3_block = bottleneck_module(conv2_2_block, lvl="2_3", pad=1, is_training=is_training, filters=256, strides=1)

            conv3_1_block = bottleneck_module(conv2_3_block, lvl="3_1", pad=1, is_training=is_training, filters=512, strides=2)
            conv3_2_block = bottleneck_module(conv3_1_block, lvl="3_2", pad=1, is_training=is_training, filters=512, strides=1)
            conv3_3_block = bottleneck_module(conv3_2_block, lvl="3_3", pad=1, is_training=is_training, filters=512, strides=1)
            conv3_4_block = bottleneck_module(conv3_3_block, lvl="3_4", pad=1, is_training=is_training, filters=512, strides=1)

            # Pad is used as dilation rate internally in bottleneck module
            conv4_1_block = bottleneck_module(conv3_4_block, lvl="4_1", pad=2, is_training=is_training, filters=1024, strides=1)
            conv4_2_block = bottleneck_module(conv4_1_block, lvl="4_2", pad=2, is_training=is_training, filters=1024, strides=1)
            conv4_3_block = bottleneck_module(conv4_2_block, lvl="4_3", pad=2, is_training=is_training, filters=1024, strides=1)
            conv4_4_block = bottleneck_module(conv4_3_block, lvl="4_4", pad=2, is_training=is_training, filters=1024, strides=1)
            conv4_5_block = bottleneck_module(conv4_4_block, lvl="4_5", pad=2, is_training=is_training, filters=1024, strides=1)
            conv4_6_block = bottleneck_module(conv4_5_block, lvl="4_6", pad=2, is_training=is_training, filters=1024, strides=1)

            conv5_1_block = bottleneck_module(conv4_6_block, lvl="5_1", pad=4, is_training=is_training, filters=2048, strides=1)
            conv5_2_block = bottleneck_module(conv5_1_block, lvl="5_2", pad=4, is_training=is_training, filters=2048, strides=1)
            conv5_3_block = bottleneck_module(conv5_2_block, lvl="5_3", pad=4, is_training=is_training, filters=2048, strides=1)

            return tf.nn.relu(conv5_3_block)

        input_shape = tf.shape(self.placeholders["image"])
        processed_image = self.placeholders["image"]

        with tf.variable_scope("PSPNet"):

            drn = build_dilated_residual_network(processed_image)

            # All conv layers from pyramid pooling module on need LR * 10 to train
            # According to original prototxt, pool sizes and strides are x10 than
            # what the paper says
            psp = pyramid_pooling_module(drn,
                                         filters=512,
                                         pool_sizes=[10, 20, 30, 60],
                                         strides_list=[10, 20, 30, 60],
                                         level_indices=[6, 3, 2, 1],
                                         is_training=self.placeholders["is_training"],
                                         name_prefix="conv5_3")

            conv5_4 = tf.layers.conv2d(psp,
                                       filters=512,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       use_bias=False,
                                       name="conv5_4")
            conv5_4_bn = tf.layers.batch_normalization(
                conv5_4, training=self.placeholders["is_training"], name="conv5_4_bn")
            conv5_4_relu = tf.nn.relu(conv5_4_bn)
            # Dropout ratio is 0.1
            conv5_4_dropout = tf.nn.dropout(
                conv5_4_relu, keep_prob=self.placeholders["keep_probability"])

            # Additional x20 to LR for conv6 according to prototxt
            conv6 = tf.layers.conv2d(conv5_4_dropout,
                                     filters=self.num_classes,
                                     kernel_size=1,
                                     strides=1,
                                     name="conv6")

            logits = tf.image.resize_bilinear(
                conv6, size=self.input_shape, align_corners=True, name="logits")

            probs = tf.nn.softmax(logits)
            output = tf.argmax(probs, axis=3, name="argmax_up")
            output = tf.expand_dims(output, axis=3, name="predict")

            return logits, probs, output
