"""
This module provides the training/testing functionality of DEC/IDEC on the
MNIST dataset.

To run DEC instead of IDEC, refer to the comment above the loss defining scope;
simply comment out the reconstruction loss Lr, and set gamma=1 in constants
"""

import os
import cv2
import sys
import logging
import datetime

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.python.framework import tensor_shape, ops
from tensorflow.python.layers.base import Layer, InputSpec

from autoencoder import build_autoencoder_graph
from utils import load_mnist, add_train_ops, cluster_accuracy, set_logging_verbosity
from constants import (
    k, Nm, gamma, tol, NUM_CHANNELS, IMAGE_SHAPE, MAX_TRAIN_ITERATIONS,
    UPDATE_P_EVERY_N_EPOCHS, TRAIN_SAVE_EVERY_N_EPOCHS, AE_LOGDIR, IDEC_LOGDIR)


def cluster_layer(inputs, n_clusters, weights=None, alpha=1.0, **kwargs):
    """Create a new clustering layer and apply it to the `inputs` tensor."""
    layer = ClusteringLayer(n_clusters, weights=weights, alpha=alpha, **kwargs)
    return layer.apply(inputs)

class ClusteringLayer(Layer):
    """
    Define a layer to calculate soft targets via Student's t-distribution.

    Input to this layer must be 2D.
    Output is a 2D tensor with shape: (None, k)
    """

    def __init__(self, k, weights=None, alpha=1.0, **kwargs):
        """Save all relevant variables needed to build the layer."""
        super(ClusteringLayer, self).__init__(**kwargs)
        self.k = k
        self.alpha = alpha
        self.initialize_with_weights = weights

        # Define an InputSpec for our layer; we don't know shape of input yet,
        # but we know it has to be 2D.
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        """
        Construct the (tensor) variables for the layer.
        This is (automatically) called a single time before the first call().
        """

        if input_shape[1].value is None or len(input_shape) != 2:
          raise ValueError('The 2nd (and last) dimension of the inputs to '
                           '`ClusteringLayer` should be defined. Found `None`.')

        # Redefine the InputSpec now that we have shape information
        self.input_spec = InputSpec(dtype=tf.float32, shape=(None, input_shape[1]))

        # Create the tensorflow variable for the trainable params of the layer
        # i.e. the weights for the similarities between embedded points and
        # cluster centroids (as measured by Student's t-distribution)
        self.clusters = self.add_variable(
            name='clusters',
            shape=[self.k, input_shape[1]],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=self.dtype,
            trainable=True)

        # If weights were provided to the constructor, load them
        if self.initialize_with_weights is not None:
            self.clusters = tf.assign(self.clusters, self.initialize_with_weights)
            del self.initialize_with_weights

        # We must assign self.built = True for tensorflow to use the layer
        self.built = True

    def call(self, inputs):
        """
        Compute soft targets q_ij via Sudent's t-distribution.

        Here we compute the numerator of equation (1) for q_ij, then normalize
        by dividing by the total sum over all vectors in the numerator.
        """

        # We use axis arg to norm so that the tensor is treated as a batch of vectors.
        num = (1.0 + tf.norm((tf.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha)
        num **= -((self.alpha + 1.0) / 2.0)
        return num / tf.reduce_sum(num)

    def compute_output_shape(self, input_shape):
        """Show output shape as (?, k)."""
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.k)

def target_distribution(q):
    """Compute the target distribution p, based on q."""
    # q in this form is a numpy array, i.e. not symbolic
    weight = q ** 2 / q.sum(axis=0)
    return (weight.T / weight.sum(axis=1)).T

def load_autoencoder_weights(sess, saver):
    """
    Initialize all variables in the session, then restore the weights of the
    autoencoder.
    """

    # Create a saver and session, then init all variables
    sess.run(tf.global_variables_initializer())

    logging.info("Attempting to restore pretrained AE weights")
    # Restore the pretrained weights of the AE
    ckpt = tf.train.get_checkpoint_state(AE_LOGDIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        logging.info("Successfully restored AE weights")
    else:
        logging.error("Unable to restore pretrained AE weights; terminating")
        sys.exit(1)

def encode_samples(samples, input_tensor, encode_fn, sess, saver):
    """
    Given a 4D tensor of samples, encode the tensor with encode_fn after
    loading the autoencoder weights.
    """

    load_autoencoder_weights(sess, saver)

    logging.info("Encoding samples into latent feature space Z")
    Z = sess.run(encode_fn, feed_dict={input_tensor: samples})

    return Z

def main():
    """Entry point."""
    set_logging_verbosity(3)

    # Make placeholder and get all hooks into the graph we'll need
    placeholders = {"image": tf.placeholder(tf.float32, shape=(None, NUM_CHANNELS, *IMAGE_SHAPE), name="image"),
                    "target": tf.placeholder(tf.float32, shape=(None, k), name="target")}

    logging.info("Building autoencoder")
    # Define the autoencoder graph, and define a saver and session
    inp, encode_fn, decode_fn = build_autoencoder_graph(placeholders)
    saver, sess = tf.train.Saver(), tf.Session()


    # Get full X and y for MNIST dataset and encode X into latent space Z
    X, y = load_mnist()
    Z = encode_samples(X, placeholders["image"], encode_fn, sess, saver)


    # Perform KMeans clustering on Z to initialize the weights of centroids
    # used in the cluster layer (i.e. the layer which computes Student's T distribution)
    logging.info("Initializing cluster centers with k-means.")
    kmeans = KMeans(n_clusters=k, n_init=20)
    y_pred = kmeans.fit_predict(Z)
    y_pred_last = np.copy(y_pred)


    logging.info("Adding clustering layer and initializing with kmeans centroids")
    cluster = cluster_layer(encode_fn, k, weights=kmeans.cluster_centers_)


    # Create loss and train ops for IDEC; set gamma=1 and comment out Lr to reduce the objective to DEC
    with tf.name_scope("loss"):
        cross_entropy = -tf.reduce_sum(placeholders["target"] * tf.log(cluster))
        entropy = -tf.reduce_sum(placeholders["target"] * tf.log(placeholders["target"] + 0.00001))
        Lc = kl_divergence = cross_entropy - entropy

        Lr = tf.losses.mean_squared_error(inp, decode_fn)

        loss = Lr + gamma * Lc


    # Add optimization ops to the graph, then reinitialize all trainable
    # variables and restore AE weights (since we reinitialized)
    trainable_var = tf.trainable_variables()
    train_op = add_train_ops(loss, trainable_var)
    load_autoencoder_weights(sess, saver)


    train_loss, index = 0, 0
    for itr in range(int(MAX_TRAIN_ITERATIONS)):
        if itr % UPDATE_P_EVERY_N_EPOCHS == 0:
            # Recompute q and p
            q = sess.run(cluster, feed_dict={placeholders["image"]: X})
            p = target_distribution(q)

            y_pred = q.argmax(1)
            if y is not None:
                acc = np.round(cluster_accuracy(y, y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                train_loss = np.round(train_loss, 5)
                logging.info(
                    "Itr={}, Acc={}, NMI={}, ARI={}, LOSS={}".format(
                        itr, acc, nmi, ari, train_loss / Nm))

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            if itr > 0 and delta_label < tol:
                logging.info(
                    "Reached tolerance threshold (delta: {} < tol: {}). "
                    "Stopping training.".format(delta_label, tol))
                break

        # train on one batch at a time
        if (index + 1) * Nm > X.shape[0]:
            feed_dict = {placeholders["image"]: X[index * Nm::],
                         placeholders["target"]: p[index * Nm::]}

            sess.run(train_op, feed_dict=feed_dict)
            train_loss = sess.run(loss, feed_dict=feed_dict)
            index = 0
        else:
            feed_dict = {placeholders["image"]: X[index * Nm:(index + 1) * Nm],
                         placeholders["target"]: p[index * Nm:(index + 1) * Nm]}

            sess.run(train_op, feed_dict=feed_dict)
            train_loss = sess.run(loss, feed_dict=feed_dict)
            index += 1


        if itr % TRAIN_SAVE_EVERY_N_EPOCHS == 0 and itr > 0:
            logging.info("Saving model at epoch: {}".format(itr))
            saver.save(sess, os.path.join(IDEC_LOGDIR, "model.ckpt"), itr)

if __name__ == "__main__":
    main()
