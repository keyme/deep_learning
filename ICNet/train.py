"""
This script provides code for training ICNet from scratch.

Paper can be found here: https://arxiv.org/pdf/1704.08545.pdf
"""

import os
import time
import logging
import datetime

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model import ICNet
from utils import (build_tfrecords_batch, create_loss, prepare_label,
                   read_and_decode_semseg_tfrecord, set_logging_verbosity)


MAX_ITERATIONS = int(1e6)
DEFAULT_NETWORK_LOCATION = "./logs_dir"

# Tensorflow cmd args
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
tf.flags.DEFINE_string("logs_dir", DEFAULT_NETWORK_LOCATION, "path to logs directory")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_integer('num_samples_per_val', '2000', "Number of samples to consider per validation period")
tf.flags.DEFINE_integer("min_after_dequeue", "100", "How big a buffer we want to sample from data")
tf.flags.DEFINE_string("path_to_records", "./tfrecords", "The path to tfrecords")
tf.flags.DEFINE_string("records_prefix", "ade20k", "The path to tfrecords")
tf.flags.DEFINE_string("dataset", "ade20k", "The path to tfrecords")

if FLAGS.dataset == "ade20k":
    # Void label is set to 0 and we have 150 classes, so we add 1 to
    # number of classes to support void label since we can't easily
    # skip it without preprocessing/remapping classes
    SEMSEG_NUM_CLASSES = 150 + 1
    SEMSEG_SHAPE = (480, 480)
    IGNORE_LABEL = 0
elif FLAGS.dataset == "cityscapes":
    SEMSEG_NUM_CLASSES = 19
    SEMSEG_SHAPE = (1024,2048)
    IGNORE_LABEL = 255
else:
    raise ValueError(
        "Unknown dataset; if training on a custom dataset, you must specify values "
        "for `SEMSEG_NUM_CLASSES`, `SEMSEG_SHAPE`, `IGNORE_LABEL`")


def main():

    set_logging_verbosity(3)

    net = ICNet(SEMSEG_SHAPE, FLAGS.logs_dir, SEMSEG_NUM_CLASSES, FLAGS.learning_rate)

    with net.graph.as_default():

        global_step = tf.Variable(0, name='global_step', trainable=False)
        best_loss = tf.Variable(1e7, name='best_loss', trainable=False, dtype=tf.float32)
        best_iou = tf.Variable(0, name='best_iou', trainable=False, dtype=tf.float32)
        best_px_acc = tf.Variable(0, name='best_px_acc', trainable=False, dtype=tf.float32)

        # Get shortcuts to all placeholders as well as the computation graph
        image = net.placeholders["image"]
        keep_prob = net.placeholders["keep_probability"]
        annotation = net.placeholders["annotation"]
        is_training = net.placeholders["is_training"]

        # Add summary ops for tensorboard
        rgb_image = tf.reverse(image, axis=[3])
        tf.summary.image("input_image", rgb_image, max_outputs=4)
        tf.summary.image(
            "ground_truth", tf.cast(tf.multiply(annotation, 255), tf.uint8), max_outputs=4)

        # Create cascade label guidence based loss expression
        lambda1, lambda2, lambda3 = 0.16, 0.4, 1.0
        sub_4_loss = create_loss(net.low_res_logits, annotation, SEMSEG_NUM_CLASSES, IGNORE_LABEL)
        sub_24_loss = create_loss(net.med_res_logits, annotation, SEMSEG_NUM_CLASSES, IGNORE_LABEL)
        sub_124_loss = create_loss(net.high_res_logits, annotation, SEMSEG_NUM_CLASSES, IGNORE_LABEL)
        loss = lambda1 * sub_4_loss + lambda2 * sub_24_loss + lambda3 * sub_124_loss
        tf.summary.scalar("entropy", loss)

        # Create expression for running mean IoU
        pred_flatten = tf.reshape(net.predict, [-1,])
        label_flatten = tf.reshape(annotation, [-1,])

        mask = tf.not_equal(label_flatten, IGNORE_LABEL)
        indices = tf.squeeze(tf.where(mask), 1)
        gt = tf.cast(tf.gather(label_flatten, indices), tf.int32)
        pred = tf.gather(pred_flatten, indices)

        mIoU, miou_update_op = tf.metrics.mean_iou(
            predictions=pred, labels=gt, num_classes=SEMSEG_NUM_CLASSES, name="mIoU")
        px_acc, px_acc_update_op = tf.metrics.accuracy(predictions=pred, labels=gt, name="px_acc")

        # Group the update ops for both metrics for convenience
        metrics_update_op = tf.group(miou_update_op, px_acc_update_op)

        # Make an operation to reset running variables for mIoU and pixel accuracy so
        # we can reset mIoU for each validation
        running_miou_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mIoU")
        running_px_acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="px_acc")

        running_metrics_init = tf.group(tf.variables_initializer(var_list=running_miou_vars),
                                        tf.variables_initializer(var_list=running_px_acc_vars))

        # Create the training and summary operations
        trainable_var = tf.trainable_variables()
        train_op = net.add_train_ops(loss, trainable_var, global_step)

        logging.info("Setting up summary op...")
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, net.sess.graph)

        # We assume a single records for train, val, test, but the
        # build_tfrecords_batch method can accept arbitrary paths
        path_to_train_tfrecord = os.path.join(
            FLAGS.path_to_records, "{}_train.tfrecords".format(FLAGS.records_prefix))
        path_to_val_tfrecord = os.path.join(
            FLAGS.path_to_records, "{}_val.tfrecords".format(FLAGS.records_prefix))
        path_to_test_tfrecord = os.path.join(
            FLAGS.path_to_records, "{}_test.tfrecords".format(FLAGS.records_prefix))

        if FLAGS.mode in ["train", "visualize"]:
            # Create data generators for train and validation
            image_batch, label_batch = build_tfrecords_batch(
                path_to_train_tfrecord, input_shape=SEMSEG_SHAPE, batch_size=FLAGS.batch_size,
                min_after_dequeue=FLAGS.min_after_dequeue, max_iterations=MAX_ITERATIONS)
            val_image_batch, val_label_batch = build_tfrecords_batch(
                path_to_val_tfrecord, input_shape=SEMSEG_SHAPE, batch_size=FLAGS.batch_size,
                min_after_dequeue=FLAGS.min_after_dequeue, max_iterations=MAX_ITERATIONS)

        # Re-run global variable initializer for the training ops just added and tfrecords
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        net.sess.run(init_op)

        # We need to reload the model since we just wiped all variables by reiniting
        net.load_network_weights(net.network_weights_loc)

        if FLAGS.mode == "train":
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=net.sess, coord=coord)

            # Begin training
            for i in range(MAX_ITERATIONS):
                itr = tf.train.global_step(net.sess, global_step)

                train_images, train_annotations = net.sess.run([image_batch, label_batch])

                feed_dict = {image: train_images, annotation: train_annotations,
                             keep_prob: 0.85, is_training: True}

                net.sess.run(train_op, feed_dict=feed_dict)

                if itr % 10 == 0 and itr > 0:
                    train_loss, summary_str = net.sess.run([loss, summary_op], feed_dict=feed_dict)

                    logging.info("Step: %d, Train_loss:%g" % (itr, train_loss))
                    summary_writer.add_summary(summary_str, itr)

                if itr % 500 == 0 and itr > 0:
                    num_batches_per_epoch = (FLAGS.num_samples_per_val // FLAGS.batch_size) + 1
                    val_losses = []

                    # Reset mIoU running vars so we get a fresh evaluation for each validation
                    net.sess.run(running_metrics_init)

                    for i in range(num_batches_per_epoch):

                        val_images, val_annotations = net.sess.run([val_image_batch, val_label_batch])
                        val_feed_dict = {image: val_images, annotation: val_annotations,
                                         keep_prob: 1.0, is_training: False}

                        val_loss, _ = net.sess.run([loss, metrics_update_op], feed_dict=val_feed_dict)
                        val_losses.append(val_loss)

                    avg_val_loss = np.mean(val_losses)
                    best_loss_val = net.sess.run(best_loss)

                    # Get current and best mean IoUs and pixel accuracies
                    val_iou, best_iou_val = net.sess.run([mIoU, best_iou])
                    val_px_acc, best_px_acc_val = net.sess.run([px_acc, best_px_acc])

                    logging.info(
                        "{} ---> Validation_loss: {} (best: {}), Validation_IoU: {} (best: {}), "
                        "Validation pixel accuracy: {} (best: {})".format(
                                    datetime.datetime.now(),
                                    avg_val_loss, best_loss_val,
                                    val_iou, best_iou_val,
                                    val_px_acc, best_px_acc_val))

                    # If this is the best we've done on validation, update checkpoint
                    if avg_val_loss < best_loss_val:
                        net.sess.run(best_loss.assign(avg_val_loss))
                        net.sess.run(best_iou.assign(val_iou))
                        net.sess.run(best_px_acc.assign(val_px_acc))
                        logging.info(
                            "New best loss: {} with iou: {} and px acc: {} at step {}".format(
                                avg_val_loss, val_iou, val_px_acc, itr))

                        net.saver.save(net.sess,
                                       os.path.join(FLAGS.logs_dir, "best_model.ckpt"),
                                       global_step=itr)

                if itr >= MAX_ITERATIONS:
                    break

            coord.request_stop()
            coord.join(threads)

        elif FLAGS.mode == "test":
            if FLAGS.dataset == "ade20k":
                # We don't have ade20k test data, so test on validation set
                test_generator = read_and_decode_semseg_tfrecord(path_to_val_tfrecord)
            else:
                test_generator = read_and_decode_semseg_tfrecord(path_to_test_tfrecord)

            all_losses = []
            for X, Y in tqdm(test_generator, desc="Testing"):
                X, Y = np.expand_dims(X, axis=0), np.expand_dims(Y, axis=0)

                feed_dict = {image: X, annotation: Y, keep_prob: 1.0, is_training: False}
                test_loss, _ = net.sess.run([loss, update_op], feed_dict=feed_dict)

                all_losses.append(test_loss)

            test_iou = net.sess.run(mIoU)
            logging.info("Test Loss: {}, mIoU: {}".format(np.mean(all_losses), test_iou))

if __name__ == "__main__":
    main()
