"""
This script demonstrates how to load the pretrained numpy weights
(we use icnet_cityscapes_trainval_90k_bnnomerge.npy specifically) from
https://github.com/hellochick/ICNet-tensorflow and use them to perform
inference with ICNet on cityscapes images.
"""

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from model import ICNet


SEMSEG_NUM_CLASSES = 19
SEMSEG_SHAPE = (1024, 2048)

BN_param_map = {'scale':    'gamma',
                'offset':   'beta',
                'variance': 'moving_variance',
                'mean':     'moving_mean'}

def main():
    """Perform inference on cityscapes data using ICNet with cityscapes weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", help="Path to cityscape .npy weights")
    parser.add_argument("input_image", help="Path to image we should predict on")
    args = parser.parse_args()

    # Create the ICNet object; learning rate does not matter here as we're not
    # training, but must be provided
    net = ICNet(SEMSEG_SHAPE, "./logs_dir", SEMSEG_NUM_CLASSES, learning_rate=1e-4)

    # Load the numpy weights; when training from scratch we generate a checkpoint
    # dir instead of saving numpy weights, but pretrained weights are in numpy
    # form, so we use those for this inference script
    data_dict = np.load(args.weights_path, encoding='latin1').item()

    for op_name in data_dict:
        for param_name, data in data_dict[op_name].items():

            if 'bn' in op_name:
                param_name = BN_param_map[param_name]

            # Map the saved tensor name to corresponding tensor in our graph
            tensor_name = "ICNet/{}/{}:0".format(op_name, param_name)
            tensor_name = tensor_name.replace('weights', 'kernel').replace('biases', 'bias')
            tensor = net.graph.get_tensor_by_name(tensor_name)

            # Set the tensor's value to the weights
            net.sess.run(tf.assign(tensor, data))

    img = cv2.imread(args.input_image)

    # Resize to input shape if necessary
    if img.shape != (*SEMSEG_SHAPE, 3):
        img = cv2.resize(img, (SEMSEG_SHAPE[1], SEMSEG_SHAPE[0]))

    feed_dict = {"input_image:0": np.expand_dims(img, axis=0),
                 "is_training:0": False,
                 "keep_probability:0": 1.0}

    pred = net.sess.run(net.output, feed_dict=feed_dict)[0]
    overlap = 0.5 * img + 0.5 * pred

    # Write output images to same directory that input came from
    output_dir = os.path.dirname(args.input_image)
    output_fname_prefix = os.path.splitext(os.path.basename(args.input_image))[0]
    cv2.imwrite("{}/{}_pred.jpg".format(output_dir, output_fname_prefix), pred)
    cv2.imwrite("{}/{}_blended.jpg".format(output_dir, output_fname_prefix), overlap)

if __name__ == "__main__":
    main()
