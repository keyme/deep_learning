"""Provide utility functions for the DEC/IDEC implementation."""
import logging

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as tf_datasets

from constants import NUM_CHANNELS, IMAGE_SHAPE


def add_train_ops(loss_val, var_list):
    """Add ops for graident descent optimization with Adam."""
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

def load_mnist():
    """
    Load the entire MNIST data (train, val, and test) from tensorflow into
    single arrays.
    """

    mnist = tf_datasets.read_data_sets("MNIST_data/")

    X_arrs, y_arrs = [], []

    # Load all MNIST data, one dataset at a time
    for dataset in [mnist.train, mnist.validation, mnist.test]:
        X, y = dataset.next_batch(dataset._num_examples, shuffle=False)

        X_arrs.append(X)
        y_arrs.append(y)

    # Stack all of the dataset matrices and reshape X into a 4D tensor
    X, y = np.concatenate(X_arrs), np.concatenate(y_arrs)
    X = X.reshape((len(X), NUM_CHANNELS, *IMAGE_SHAPE))

    # Shuffle full arrays in tandum
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)

    return X, y

def compute_total_num_params():
    """Compute the total number of training params in the default tf graph."""
    total_parameters = 0
    for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def cluster_accuracy(y_true, y_pred):
    """Calculate the scaler clustering accuracy."""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def set_logging_verbosity(verbose_level=3):
    """Set the level of logging verbosity."""

    if not isinstance(verbose_level, int):
        raise TypeError("verbose_level must be an int")

    if not (0 <= verbose_level <= 4):
        raise ValueError("verbose_level must be between 0 and 4")

    verbosity = [logging.CRITICAL,
                 logging.ERROR,
                 logging.WARNING,
                 logging.INFO,
                 logging.DEBUG]

    logging.basicConfig(
        format='%(asctime)s:\t %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=verbosity[verbose_level])
