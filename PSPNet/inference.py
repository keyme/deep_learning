"""
This module uses pretrained PSPNet50 for ADE20k to predict on a provided image.
"""

import argparse
import h5py
import numpy as np
import os
from scipy import misc, ndimage
import tensorflow as tf

from model import PSPNet50
from utils import set_logging_verbosity, add_color


SEMSEG_SHAPE = (473, 473)
SEMSEG_NUM_CLASSES = 150
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])


def main():
    """Predict with ADE20K PSPNet50."""
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", help="Path to ade20k h5 weights")
    parser.add_argument("input_image", help="Path to image we should predict on")
    args = parser.parse_args()

    # We're not training so provided learning_rate doesn't matter
    net = PSPNet50(SEMSEG_SHAPE, "pspnet50_ade20k", SEMSEG_NUM_CLASSES, learning_rate=1e-4)

    with net.graph.as_default():
        output_dir = os.path.dirname(args.input_image)
        input_fname = os.path.splitext(os.path.basename(args.input_image))[0]

        # Read the input image and perform ADE20K preprcoessing
        img = misc.imread(args.input_image, mode='RGB')
        h_ori, w_ori = img.shape[:2]
        img = misc.imresize(img, SEMSEG_SHAPE)

        img = img - DATA_MEAN
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.astype('float32')

        # Load the weights from h5 file and assign them to corresponding tf tensors
        weights = h5py.File(args.weights_path, 'r')
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            layer_name, tensor_name = i.name.split('/')[-2], i.name.split('/')[-1]

            t = net.graph.get_tensor_by_name(i.name)

            load_weights = weights[layer_name][layer_name][tensor_name].value
            if tensor_name in ["gamma:0", "beta:0", "moving_mean:0", "moving_variance:0"]:
                load_weights = load_weights.reshape(-1)

            net.sess.run(tf.assign(t, load_weights))

        # Predict on the image
        feed_dict = {"input_image:0": np.expand_dims(img, axis=0),
                     "is_training:0": False,
                     "keep_probability:0": 1.0}
        pred = net.sess.run(net.logits, feed_dict=feed_dict)[0]

        # If predict shape isn't the same as original shape, make predict shape the same
        if img.shape[0:1] != SEMSEG_SHAPE[0]:
            h, w = pred.shape[:2]
            pred = ndimage.zoom(pred, (1. * h_ori / h, 1. * w_ori / w, 1.),
                                 order=1, prefilter=False)

        cm = np.argmax(pred, axis=2)

        # Create human digestable prediction images
        color_cm = add_color(cm)
        alpha_blended = 0.5 * color_cm * 255 + 0.5 * misc.imread(args.input_image, mode='RGB')
        color_path = os.path.join(output_dir, "{}_colored_pred.jpg".format(input_fname))
        blend_path = os.path.join(output_dir, "{}_blended_pred.jpg".format(input_fname))
        misc.imsave(color_path, color_cm)
        misc.imsave(blend_path, alpha_blended)

if __name__ == "__main__":
    main()
