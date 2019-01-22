"""
This module provides an implemenation of ICNet. Specifically, the architecture
of this ICNet implementation is a copy of the architecture given by the
cityscape model's prototxt by the original author.
(https://github.com/hszhao/ICNet/blob/master/evaluation/prototxt/icnet_cityscapes_bnnomerge.prototxt).

Special thanks to https://github.com/hellochick/ICNet-tensorflow;
we use their decode/coloring functions and some code pertaining to training/inference,
as well as using their cityscape weights to verify correctness.

Author: Nick Marton
"""

import numpy as np
import tensorflow as tf

from utils import decode_labels, zero_padding


def cascade_feature_fusion_module(f1, f2, c3, is_training, names):
    """
    Perform cascade feature fusion between f1 and f2.

    `names` argument is only needed to match names of pretrained weights.
    """

    # f2 height and width should always be double that of f1's
    f2_shape = tf.shape(f2)
    f1_interp = tf.image.resize_bilinear(
        f1, f2_shape[1:-1], align_corners=True)


    f1_padded = zero_padding(f1_interp, paddings=2)
    f1_conv = tf.layers.conv2d(
        f1_padded, kernel_size=3, strides=1, filters=c3,
        dilation_rate=2, use_bias=False, name=names["f1_conv"])
    f1_bn = tf.layers.batch_normalization(
        f1_conv, momentum=0.95, epsilon=1e-5,
        training=is_training, name=names["f1_bn"])


    f2_proj = tf.layers.conv2d(
        f2, filters=c3, kernel_size=1, strides=1,
        use_bias=False, name=names["f2_conv"])
    f2_bn = tf.layers.batch_normalization(
        f2_proj, momentum=0.95, epsilon=1e-5,
        training=is_training, name=names["f2_bn"])


    cff = tf.add_n([f2_bn, f1_bn], name=names["out"])

    # We need a hook into f1_interp to create sum{4,24}_outs;
    # return it as well
    return f1_interp, tf.nn.relu(cff)

def pyramid_pooling_module(input_layer, filters, pool_sizes, strides_list,
                           level_indices, is_training, name_prefix, convolve=False):
    """
    Pyramid pooling module.

    This module will create a level for every triplet in
    `zip(pool_sizes, strides_list, level_indices)`.

    `level_indices` is only used in the naming of layers in each level and is
    necessary to keep this module arbitrary but still allow for original
    paper's caffe converted weights to be loaded.

    This module assumes data_format is `channels_last`.

    Original PSPNet paper convolves the output of the pooling layer
    (in addition to adding batch norm and relu), however, ICNet cityscape
    prototxt by the author skips these layers and just interpolates the pooling
    output. Hence, we provide the option `convolve` to support both of these
    cases. We default convolve to `False` as to ensure inference is performed
    correctly after loading pretrained weights.
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

        if convolve:
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

            prev = level_interp
        else:
            prev = level_pool

        level_interp = tf.image.resize_bilinear(
            prev, input_shape[1:-1], align_corners=True,
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

        elesum = tf.add_n(concat_layers)

    return elesum


class ICNet(object):
    """Implement abstract base for ICNet architecture."""
    def __init__(self, input_shape, network_storage_dir, num_classes, learning_rate,
                 is_grayscale=False):
        """Construct the ICNet object."""
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
            image = tf.placeholder(
                tf.float32, shape=[None, input_shape[0], input_shape[1], self.num_channels],
                name="input_image")
            annotation = tf.placeholder(
                tf.int32, shape=[None, input_shape[0], input_shape[1], 1],
                name="annotation")
            is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
            # We don't use dropout in the cityscapes/ade20k architecture,
            # but leave the placeholder as our internal version uses dropout
            keep_probability = tf.placeholder(tf.float32, name="keep_probability")

            self.placeholders = {"image": image,
                                 "keep_probability": keep_probability,
                                 "annotation": annotation,
                                 "is_training": is_training}

            # Build the computational graph
            hooks = self.build_computational_graph()
            self.low_res_logits, self.med_res_logits, self.high_res_logits, self.predict, self.output = hooks

            # Set up the saver and try to restore any model previously being trained
            self.saver = tf.train.Saver()

    def build_computational_graph(self):
        """Build the ICNet with ResNet50 backbone. and 4 level PSP module."""

        def bottleneck_module(inputs, lvl, pad, is_training, filters, strides,
                              data_format='channels_last', bottleneck_factor=4):
            """
            Implement the bottleneck module proposed in ResNet.
            1x1 conv -> 3x3 conv -> 1x1 conv
            """

            # 1x1 reduce component
            x = tf.layers.conv2d(inputs, filters=filters // bottleneck_factor,
                                 kernel_size=1, strides=strides, data_format=data_format,
                                 use_bias=False, name="conv{}_1x1_reduce".format(lvl))
            x = tf.layers.batch_normalization(
                x, momentum=0.95, epsilon=1e-5, training=is_training,
                name="conv{}_1x1_reduce_bn".format(lvl))
            x = tf.nn.relu(x)

            # 3x3 component
            x = zero_padding(x, pad)
            x = tf.layers.conv2d(x, filters=filters // bottleneck_factor,
                                 kernel_size=3, strides=1, dilation_rate=pad,
                                 data_format=data_format, use_bias=False,
                                 name="conv{}_3x3".format(lvl))
            x = tf.layers.batch_normalization(
                x, momentum=0.95, epsilon=1e-5, training=is_training,
                name="conv{}_3x3_bn".format(lvl))
            x = tf.nn.relu(x)

            # 1x1 increase component
            x = tf.layers.conv2d(x, filters=filters, kernel_size=1, strides=1,
                                 data_format=data_format, use_bias=False,
                                 name="conv{}_1x1_increase".format(lvl))
            x = tf.layers.batch_normalization(
                x, momentum=0.95, epsilon=1e-5, training=is_training,
                name="conv{}_1x1_increase_bn".format(lvl))

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
                    conv_proj, momentum=0.95, epsilon=1e-5, training=is_training,
                    name="conv{}_1x1_proj_bn".format(lvl))
                out = x + conv_proj_bn
            else:
                out = x + inputs

            return tf.nn.relu(out)

        def build_dilated_residual_network(input_layer):
            """Construct a 34-layer variant dilated residual network."""
            is_training = self.placeholders["is_training"]

            conv1_1 = tf.layers.conv2d(
                input_layer, filters=32, kernel_size=3, strides=2,
                padding="same", use_bias=False, name="conv1_1_3x3_s2")
            conv1_1_bn = tf.layers.batch_normalization(
                conv1_1, momentum=0.95, epsilon=1e-5, training=is_training, name="conv1_1_3x3_s2_bn")
            conv1_1_relu = tf.nn.relu(conv1_1_bn)

            conv1_2 = tf.layers.conv2d(
                conv1_1_relu, filters=32, kernel_size=3, strides=1,
                padding="same", use_bias=False, name="conv1_2_3x3")
            conv1_2_bn = tf.layers.batch_normalization(
                conv1_2, momentum=0.95, epsilon=1e-5, training=is_training, name="conv1_2_3x3_bn")
            conv1_2_relu = tf.nn.relu(conv1_2_bn)

            conv1_3 = tf.layers.conv2d(
                conv1_2_relu, filters=64, kernel_size=3, strides=1,
                padding="same", use_bias=False, name="conv1_3_3x3")
            conv1_3_bn = tf.layers.batch_normalization(
                conv1_3, momentum=0.95, epsilon=1e-5, training=is_training, name="conv1_3_3x3_bn")
            conv1_3_relu = tf.nn.relu(conv1_3_bn)

            padding0 = zero_padding(conv1_3_relu, paddings=1)
            pool1 = tf.layers.max_pooling2d(
                padding0, pool_size=3, strides=2, padding='valid', name="pool1")

            conv2_1_block = bottleneck_module(pool1, lvl="2_1", pad=1, is_training=is_training, filters=128, strides=1)
            conv2_2_block = bottleneck_module(conv2_1_block, lvl="2_2", pad=1, is_training=is_training, filters=128, strides=1)
            conv2_3_block = bottleneck_module(conv2_2_block, lvl="2_3", pad=1, is_training=is_training, filters=128, strides=1)

            conv3_1_block = bottleneck_module(conv2_3_block, lvl="3_1", pad=1, is_training=is_training, filters=256, strides=2)

            # We share weights for the low and med resolution levels;
            # conv3_1_sub4 is a hook into the end of med resolution level
            conv3_1_sub4 = tf.image.resize_bilinear(
                conv3_1_block, tf.shape(conv3_1_block)[1:-1] // 2,
                align_corners=True, name="conv3_1_sub4")

            conv3_2_block = bottleneck_module(conv3_1_sub4, lvl="3_2", pad=1, is_training=is_training, filters=256, strides=1)
            conv3_3_block = bottleneck_module(conv3_2_block, lvl="3_3", pad=1, is_training=is_training, filters=256, strides=1)
            conv3_4_block = bottleneck_module(conv3_3_block, lvl="3_4", pad=1, is_training=is_training, filters=256, strides=1)

            # Pad is used as dilation rate internally in bottleneck module
            conv4_1_block = bottleneck_module(conv3_4_block, lvl="4_1", pad=2, is_training=is_training, filters=512, strides=1)
            conv4_2_block = bottleneck_module(conv4_1_block, lvl="4_2", pad=2, is_training=is_training, filters=512, strides=1)
            conv4_3_block = bottleneck_module(conv4_2_block, lvl="4_3", pad=2, is_training=is_training, filters=512, strides=1)
            conv4_4_block = bottleneck_module(conv4_3_block, lvl="4_4", pad=2, is_training=is_training, filters=512, strides=1)
            conv4_5_block = bottleneck_module(conv4_4_block, lvl="4_5", pad=2, is_training=is_training, filters=512, strides=1)
            conv4_6_block = bottleneck_module(conv4_5_block, lvl="4_6", pad=2, is_training=is_training, filters=512, strides=1)

            conv5_1_block = bottleneck_module(conv4_6_block, lvl="5_1", pad=4, is_training=is_training, filters=1024, strides=1)
            conv5_2_block = bottleneck_module(conv5_1_block, lvl="5_2", pad=4, is_training=is_training, filters=1024, strides=1)
            conv5_3_block = bottleneck_module(conv5_2_block, lvl="5_3", pad=4, is_training=is_training, filters=1024, strides=1)

            return conv3_1_block, conv5_3_block

        input_shape = tf.shape(self.placeholders["image"])
        processed_image = self.placeholders["image"]
        is_training = self.placeholders["is_training"]

        with tf.variable_scope("ICNet"):

            # Assume NHWC
            data_sub2 = tf.image.resize_bilinear(processed_image,
                                                 (input_shape[1:-1] // 2),
                                                 align_corners=True,
                                                 name="data_sub2")

            conv3_1, drn = build_dilated_residual_network(data_sub2)

            # According to paper: "1/4 sized image is fed into PSPNet with
            # downsampling rate 8, resulting in a 1/32-resolution feature map".
            # However, according to author's cityscape prototxt, we feed a
            # 1/2 sized image into PSPNet with downsampling rate 16. Either way
            # results in 1/32 resolution feature map; compute those dimensions
            h, w = self.input_shape[0] // 32, self.input_shape[1] // 32
            pool_sizes = strides_list = [
                (h, w), (h / 2, w / 2), (h / 3, w / 3), (h / 4, w / 4)]
            # These are used to match names pretrained weights
            level_indices = [1, 2, 3, 6]
            psp = pyramid_pooling_module(drn,
                                         filters=256,
                                         pool_sizes=pool_sizes,
                                         strides_list=strides_list,
                                         level_indices=level_indices,
                                         is_training=is_training,
                                         name_prefix="conv5_3",
                                         convolve=False)

            conv5_4 = tf.layers.conv2d(psp,
                                       filters=256,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       use_bias=False,
                                       name="conv5_4_k1")
            conv5_4_bn = tf.layers.batch_normalization(
                conv5_4, momentum=0.95, epsilon=1e-5,
                training=is_training, name="conv5_4_k1_bn")
            conv5_4_bn = tf.nn.relu(conv5_4_bn)


            # Build light high resolution CNN on top of input
            conv1_sub1 = tf.layers.conv2d(
                processed_image, kernel_size=3, filters=32, strides=2,
                padding="same", use_bias=False, name="conv1_sub1")
            conv1_sub1_bn = tf.layers.batch_normalization(
                conv1_sub1, momentum=0.95, epsilon=1e-5,
                training=is_training, name="conv1_sub1_bn")
            conv1_sub1_relu = tf.nn.relu(conv1_sub1_bn)

            conv2_sub1 = tf.layers.conv2d(
                conv1_sub1_relu, kernel_size=3, filters=32, strides=2,
                padding="same", use_bias=False, name="conv2_sub1")
            conv2_sub1_bn = tf.layers.batch_normalization(
                conv2_sub1, momentum=0.95, epsilon=1e-5,
                training=is_training, name="conv2_sub1_bn")
            conv2_sub1_relu = tf.nn.relu(conv2_sub1_bn)

            conv3_sub1 = tf.layers.conv2d(
                conv2_sub1_relu, kernel_size=3, filters=64, strides=2,
                padding="same", use_bias=False, name="conv3_sub1")
            conv3_sub1_bn = tf.layers.batch_normalization(
                conv3_sub1, momentum=0.95, epsilon=1e-5,
                training=is_training, name="conv3_sub1_bn")
            conv3_sub1_relu = tf.nn.relu(conv3_sub1_bn)


            # Do cascade feature fusion for sub24
            sub24_cff_names = {"f1_conv": "conv_sub4",
                               "f1_bn": "conv_sub4_bn",
                               "f2_conv": "conv3_1_sub2_proj",
                               "f2_bn": "conv3_1_sub2_proj_bn",
                               "out": "sub24_sum"}
            conv5_4_interp, sub24_sum_relu = cascade_feature_fusion_module(
                f1=conv5_4_bn, f2=conv3_1, c3=128,
                is_training=is_training, names=sub24_cff_names)


            # Do cascade feature fusion for sub12
            sub12_cff_names = {"f1_conv": "conv_sub2",
                               "f1_bn": "conv_sub2_bn",
                               "f2_conv": "conv3_sub1_proj",
                               "f2_bn": "conv3_sub1_proj_bn",
                               "out": "sub12_sum"}
            sub24_sum_interp, sub12_sum_relu = cascade_feature_fusion_module(
                f1=sub24_sum_relu, f2=conv3_sub1_relu, c3=128,
                is_training=is_training, names=sub12_cff_names)


            # Get the sub outputs to use in cascade label guidance
            low_res_logits = tf.layers.conv2d(
                conv5_4_interp, kernel_size=1, filters=self.num_classes,
                strides=1, name="sub4_out")
            med_res_logits = tf.layers.conv2d(
                sub24_sum_interp, kernel_size=1, filters=self.num_classes,
                strides=1, name="sub24_out")

            # interpolate to output feature map size (1/4 input) and project to
            # final number of classes to get logits
            output_shape = (self.input_shape[0] // 4, self.input_shape[1] // 4)
            sub12_sum_interp = tf.image.resize_bilinear(
                sub12_sum_relu, output_shape, name="sub12_sum_interp")
            conv6_cls = tf.layers.conv2d(
                sub12_sum_interp, kernel_size=1, filters=self.num_classes,
                strides=1, name="conv6_cls")

            high_res_logits = conv6_cls

            # Upscale the logits and decode prediction to get final result.
            logits_up = tf.image.resize_bilinear(
                high_res_logits, size=self.input_shape, align_corners=True)
            logits_up_cropped = tf.image.crop_to_bounding_box(
                logits_up, 0, 0, self.input_shape[0], self.input_shape[1])

            # Create output node for evaluation
            raw_predict = tf.argmax(logits_up, axis=3)
            predict = tf.expand_dims(raw_predict, axis=3, name="predict")

            # Create output node for inference
            output_classes = tf.argmax(logits_up_cropped, axis=3)
            output = decode_labels(output_classes, self.input_shape, self.num_classes)

            return low_res_logits, med_res_logits, high_res_logits, predict, output

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
