"""Define and pretrain an autoencoder as defined in the DEC/IDEC papers."""

import os
import sys
import logging
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as tf_datasets

from constants import (k, Nm, NUM_CHANNELS, IMAGE_SHAPE,
                       PRETRAIN_SAVE_EVERY_N_EPOCHS, PRETRAIN_ITERATIONS, AE_LOGDIR)
from utils import load_mnist, add_train_ops, set_logging_verbosity


def pretrain_autoencoder(placeholders):
    """Pretrain the autoencoder end-to-end on MSE objective."""
    inp, encode_fn, decode_fn = build_autoencoder_graph(placeholders)

    saver = tf.train.Saver()
    sess = tf.Session()

    # AE loss is just MSE between (flattened) input and decoder output
    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(inp, decode_fn)

    # Collect trainable variables, define train op as Adam and init the session
    trainable_var = tf.trainable_variables()
    train_op = add_train_ops(loss, trainable_var)

    # Init graph after adding train/loss ops
    sess.run(tf.global_variables_initializer())

    X, _ = load_mnist()

    num_batches = len(X) // Nm
    for epoch in range(PRETRAIN_ITERATIONS):

        # Shuffle X before each epoch
        np.random.shuffle(X)

        # Train over all full batches; we shuffle so we don't have to worry
        # about when a partial batch is left over
        train_loss = 0
        for batch_ix in range(num_batches):
            # Get next batch for training (4D tensor is returned;
            # our first layer after the input layer will flatten the input)
            batch_X = X[batch_ix * Nm: (batch_ix + 1) * Nm]

            # Train on the batch and accumulate the loss to find avg batch loss
            _, batch_loss = sess.run(
                [train_op, loss], feed_dict={placeholders["image"]: batch_X})
            train_loss += batch_loss

        logging.info(
            "{}\t Epoch: {}, Train_loss: {}".format(
                datetime.datetime.now(), epoch, train_loss / num_batches))

        # Save the model every n epochs
        if epoch % PRETRAIN_SAVE_EVERY_N_EPOCHS == 0 and epoch > 0:
            saver.save(sess, os.path.join(AE_LOGDIR, "model.ckpt"), epoch)

def build_autoencoder_graph(placeholders):
    """Construct the autoencoder."""
    image = placeholders["image"]

    # Flatten input image before dense layer so our tensor shapes are nice
    flat = tf.contrib.layers.flatten(image)
    encoder_0 = tf.layers.dense(flat, units=500, activation=tf.nn.relu, name="encoder_0")
    encoder_1 = tf.layers.dense(encoder_0, units=500, activation=tf.nn.relu, name="encoder_1")
    encoder_2 = tf.layers.dense(encoder_1, units=2000, activation=tf.nn.relu, name="encoder_2")
    # As specified in paper, do not apply relu to last encoding layer
    encoder_3 = tf.layers.dense(encoder_2, units=k, activation=None, name="encoder_3")
    decoder_3 = tf.layers.dense(encoder_3, units=2000, activation=tf.nn.relu, name="decoder_3")
    decoder_2 = tf.layers.dense(decoder_3, units=500, activation=tf.nn.relu, name="decoder_2")
    decoder_1 = tf.layers.dense(decoder_2, units=500, activation=tf.nn.relu, name="decoder_1")
    # As specified in paper, do not apply relu to last decoding layer
    decoder_0 = tf.layers.dense(decoder_1, units=flat.shape[-1], activation=None, name="decoder_0")

    return flat, encoder_3, decoder_0

def main():
    """Entry point."""
    set_logging_verbosity(3)

    placeholders = {
    "image": tf.placeholder(
        tf.float32, shape=(None, NUM_CHANNELS, *IMAGE_SHAPE), name="image")}

    pretrain_autoencoder(placeholders)

if __name__ == "__main__":
    main()
