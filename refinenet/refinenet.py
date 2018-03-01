"""Contains implementation of RefineNet on ResNet-50
"""

import numpy as np
import tensorflow as tf

from resnet.resnet_v2 import resnet_v2_50 as resnet_50

NUM_REFINE_FILTERS = 256
INPUT_IMAGE_SHAPE = (512, 512)

class RefineNet(object):
    def __init__(self,
                 input_shape,
                 is_training,
                 num_classes,
                 learning_rate=1e-4,
                 network_storage_dir=None):
        """Construct the RefineNet object."""
        self.input_shape = input_shape
        self.is_training = is_training
        self.network_weights_loc = network_storage_dir
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.data_format = "channels_last"

        tf.reset_default_graph()

        # Set up the placeholder tensors to use as input/output to network
        image = tf.placeholder(
            tf.float32, shape=[None, input_shape[0], input_shape[1], 3],
            name="input_image")
        annotation = tf.placeholder(
            tf.int32, shape=[None, input_shape[0], input_shape[1], 1],
            name="annotation")

        self.placeholders = {"image": image,
                             "annotation": annotation}

        # Build the computational graph
        self.predict_with_network, self.logits = self.build_computational_graph()

        self.loss = tf.reduce_mean(
                       (tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.logits, labels=tf.squeeze(annotation, squeeze_dims=[3]), name="xentropy")))

        # get train op
        trainable_var = tf.trainable_variables()
        self.train_op = self.train(self.loss, trainable_var)

        self.num_params = self.compute_total_num_params()

        self.sess = tf.Session()

        self.saver = tf.train.Saver()

        # Initialize the graph, then load the network into memory.
        self.sess.run(tf.global_variables_initializer())
        if network_storage_dir:
            self.load_network_weights(self.network_weights_loc)

    def load_network_weights(self, path_to_network):
        """Load the network weights contained in `path_to_network` into memory."""
        ckpt = tf.train.get_checkpoint_state(path_to_network)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.weights_loaded = True
        else:
            self.weights_loaded = False

    def compute_total_num_params(self):
        """Compute the number of parameters of this model"""
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def build_computational_graph(self):
        """Build the semantic segmentation network's computational graph."""

        processed_image = self.placeholders["image"]

        if self.data_format == "channels_first":
            processed_image = tf.transpose(processed_image, [0, 3, 1, 2])

        num_filters = NUM_REFINE_FILTERS

        def residual_conv_unit(x):
            """Residual Convolutional Unit"""
            # ReLU -> 3x3 Conv -> ReLU -> 3x3 Conv
            relu = tf.nn.relu(x)
            conv = tf.layers.conv2d(relu, filters=num_filters,
                                    kernel_size=3, strides=1, padding="same",
                                    data_format=self.data_format)
            relu = tf.nn.relu(conv)
            conv = tf.layers.conv2d(relu, filters=num_filters,
                                    kernel_size=3, strides=1, padding="same",
                                    data_format=self.data_format)

            # residual connection
            compare_dimension = 1 if self.data_format == "channels_first" else 3
            if x.get_shape()[compare_dimension] == conv.get_shape()[compare_dimension]:
                # if same dimension, then add them up
                out = tf.add(x, conv)
            else:
                # if different dimension, then project shortcut x to have the same dimension with conv
                # because the number of channels can be different for x and conv
                x_conv = tf.layers.conv2d(x, filters=num_filters,
                                          kernel_size=1, strides=1, padding="same",
                                          data_format=self.data_format)
                out = tf.add(x_conv, conv)
            return out

        def multi_resolution_fusion(low_res_input, high_res_input):
            """Multi-Resolution Fusion
               we assume both the width and height of low_res_out are half of high_res_out
            """
            def conv_and_upsample(x, stride):
                """Helper function to create (Conv->Deconv)"""
                conv = tf.layers.conv2d(x, filters=num_filters, kernel_size=3,
                                        strides=stride, padding="same",
                                        data_format=self.data_format)
                # transposed convolutional layer to upsample the feature map
                upsample = tf.layers.conv2d_transpose(conv, filters=num_filters, kernel_size=3,
                                                      strides=2, padding="same",
                                                      data_format=self.data_format)
                return upsample
            if low_res_input is not None:
                # we upsample the low resolution input to have the same dimension of high resolution input
                low_res_out = conv_and_upsample(low_res_input, stride=1)
            high_res_out = conv_and_upsample(high_res_input, stride=2)
            if low_res_input is not None:
                return tf.add(low_res_out, high_res_out)
            else:
                return high_res_out

        def chained_residual_pooling(x):
            """Chained_Residual_Pooling"""
            def pool_and_conv(x):
                """Help function to create (Max_pool->Conv)"""
                ksize = [1, 1, 5, 5] if self.data_format == "channels_first" else [1, 5, 5, 1]
                data_format = "NCHW" if self.data_format == "channels_first" else "NHWC"
                max_pool = tf.nn.max_pool(x, ksize=ksize, strides=[1, 1, 1, 1], padding="SAME", data_format=data_format)
                conv = tf.layers.conv2d(max_pool, filters=num_filters,
                                        kernel_size=3, strides=1, padding="same",
                                        data_format=self.data_format)
                return conv

            relu = tf.nn.relu(x)
            pool_conv_out = relu
            sum_out = relu
            # the number of (max_pool->conv) is a design choice
            # change num_pool_conv_units to suit your own model
            num_pool_conv_units = 2
            for i in range(num_pool_conv_units):
                pool_conv_out = pool_and_conv(pool_conv_out)
                sum_out = tf.add(sum_out, pool_conv_out)
            return sum_out

        def refine_block(low_res_input, high_res_input):
            """ RefineNet Block """
            # Adaptive Conv
            low_res_adap_conv_out = None
            if low_res_input is not None:
                low_res_adap_conv_out = residual_conv_unit(low_res_input)
                low_res_adap_conv_out = residual_conv_unit(low_res_adap_conv_out)
            high_res_adap_conv_out = residual_conv_unit(high_res_input)
            high_res_adap_conv_out = residual_conv_unit(high_res_adap_conv_out)

            # Multi-resolution Fusion
            mrf_out = multi_resolution_fusion(low_res_adap_conv_out, high_res_adap_conv_out)

            # Chained Residual Pooling
            crp_out = chained_residual_pooling(mrf_out)

            out_conv = residual_conv_unit(crp_out)

            return out_conv


        # build ResNet
        nets, end_points = resnet_50(inputs=processed_image,
                                     num_classes=self.num_classes,
                                     is_training=self.is_training,
                                     scope="resnet_v2")

        # get outputs from ResNet that we want to refine on
        multi_res_outs = [end_points['resnet_v2/block1/unit_2/bottleneck_v2'],
                          end_points['resnet_v2/block2/unit_3/bottleneck_v2'],
                          end_points['resnet_v2/block3/unit_5/bottleneck_v2'],
                          end_points['resnet_v2/block4']]

        # refine outputs recursively
        refine_out = None
        
        # refine feature maps recursively
        # start from the highest-level feature maps
        # we need to refine 4 feature maps in total
        for i in [3, 2, 1, 0]:
            refine_out = refine_block(refine_out, multi_res_outs[i])


        # two more rcus right before the final scoring layer
        for i in range(2):        
            refine_out = residual_conv_unit(refine_out)
        
        # scoring layer
        logits = tf.layers.conv2d(refine_out, filters=self.num_classes,
                                      kernel_size=3, strides=1, padding="same",
                                      data_format=self.data_format)

        if self.data_format == "channels_first":
            logits = tf.transpose(logits, [0, 2, 3, 1])

        # bilinearly up-sample to match the dimension of the input image
        logits = tf.image.resize_images(logits, self.input_shape)

        annotation_pred = tf.argmax(logits, axis=3, name="prediction", output_type=tf.int32)

        return tf.expand_dims(annotation_pred, dim=3), logits

    def train(self, loss_val, var_list):
        """Add training operations to computational graph."""
        self.global_step = tf.get_variable(name="global_step", dtype=tf.int32, initializer=0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads = optimizer.compute_gradients(loss_val, var_list=var_list)
            return optimizer.apply_gradients(grads, self.global_step)

if __name__ == "__main__":
    """ run this main function to test the model """

    # build RefineNet
    model = RefineNet(input_shape=INPUT_IMAGE_SHAPE,
                      is_training=True,
                      num_classes=2,
                      learning_rate=1e-4)

    print("RefineNet-ResNet-50 has {} parameters.".format(model.num_params))

    
    # generate a random image
    random_image = np.random.rand(1, INPUT_IMAGE_SHAPE[0], INPUT_IMAGE_SHAPE[1], 3)
    print("image shape: ", random_image.shape)

    # generate a random label (0 or 1) for each pixel
    random_annotation = np.random.randint(0, 2, (1, INPUT_IMAGE_SHAPE[0], INPUT_IMAGE_SHAPE[1], 1))
    print("annotation shape: ", random_annotation.shape)

    # train for one iteration
    loss_val, _ = model.sess.run([model.loss, model.train_op], 
                   feed_dict={model.placeholders["image"]: random_image, 
                              model.placeholders["annotation"]: random_annotation})
    
    # check results
    print("loss_val: ", loss_val)

    print('test finished.')
