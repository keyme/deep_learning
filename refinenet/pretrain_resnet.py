"""
This script trains ResNet-50 on a classification task, Tiny ImageNet dataset (200 classes)
Use wrangle_tiny_imagenet.py to prepare the raw Tiny ImageNet dataset for this script
"""
import os
import datetime
import argparse

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from resnet.resnet_v2 import resnet_v2_50 as resnet_50

def pretrain_resnet(logdir,
                    path_to_imagenet_data,
                    num_classes=200,
                    input_shape=(256, 256),
                    learning_rate=1e-4,
                    batch_size=32,
                    max_itrs=1e5,
                    print_train_every_itrs=20,
                    validation_every_itrs=200):
    """Pre-train the resnet componenet of RefineNet-ResNet on ImageNet dat
    """
    # create logdir if its not there
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    tf.reset_default_graph()

    # placeholders for images and their labels
    input_images = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], 3], name="resnet_input_images")
    labels = tf.placeholder(tf.int32, shape=[None], name="resnet_labels")

    # build ResNet
    resnet_logits, end_points = resnet_50(inputs=input_images,
                                          num_classes=num_classes,
                                          is_training=True,
                                          scope="resnet_v2")

    # top-1 error
    t1_err = 1 - tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=resnet_logits, targets=labels, k=1), tf.float32))

    # top-5 error
    t5_err = 1 - tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=resnet_logits, targets=labels, k=5), tf.float32))

    # compute loss for classification task
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                           logits=resnet_logits, labels=labels, name="resnet_entropy")))

    # global step for training
    global_step = tf.get_variable("resnet_global_step", dtype=tf.int32, initializer=0)

    sess = tf.Session()

    # get train step
    train_op = tf.train.AdamOptimizer(learning_rate, name='resnet_Adam').minimize(loss, global_step=global_step)

    # initialize all variables
    sess.run(tf.global_variables_initializer())

    # restore weights if logdir has any checkpoints
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # tensorboard related
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('top1 error', t1_err)
    tf.summary.scalar('top5 error', t5_err)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logdir+'/train', sess.graph)
    val_writer = tf.summary.FileWriter(logdir+'/val', sess.graph)

    # prepare data
    tiny_imagenet_classes = []
    with open(os.path.join(path_to_imagenet_data, 'wnids.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            wnid = line.strip()
            tiny_imagenet_classes.append(wnid)

    train_datagen = ImageDataGenerator(samplewise_center=True)
    train_datagen = train_datagen.flow_from_directory(os.path.join(path_to_imagenet_data, 'train_keras'),
                                                      classes=tiny_imagenet_classes,
                                                      class_mode='sparse',
                                                      target_size=input_shape,
                                                      batch_size=batch_size)

    val_datagen = ImageDataGenerator(samplewise_center=True)
    val_datagen = val_datagen.flow_from_directory(os.path.join(path_to_imagenet_data, 'val_keras'),
                                                  classes=tiny_imagenet_classes,
                                                  class_mode='sparse',
                                                  target_size=input_shape,
                                                  batch_size=batch_size)

    # training loop
    best_val_error = float('inf')
    total_t1_err_val, total_t5_err_val = 0, 0
    for itr, batch_data in enumerate(train_datagen):
        if itr >= max_itrs:
            break
        
        # train step
        X_train_batch, y_train_batch = batch_data
        summary, loss_val, t1_err_val, t5_err_val, _ = sess.run([merged, loss, t1_err, t5_err, train_op],
                                                       feed_dict={input_images: X_train_batch, labels: y_train_batch})
        g_val = sess.run(global_step) - 1
        train_writer.add_summary(summary, g_val)

        if itr % print_train_every_itrs == 0:
            print("Itr {}: training loss {}, top-1 error {}, top-5 error {}".format(itr, loss_val, t1_err_val, t5_err_val))
        
        # validation
        X_val_batch, y_val_batch = next(val_datagen)
        summary, t1_err_val, t5_err_val = sess.run([merged, t1_err, t5_err], feed_dict={input_images: X_val_batch, labels: y_val_batch})
        val_writer.add_summary(summary, g_val)
        total_t1_err_val += t1_err_val
        total_t5_err_val += t5_err_val
        
        if itr % validation_every_itrs == 0:
            # aggregate validation stats
            total_t1_err_val /= min(itr+1, validation_every_itrs)
            total_t5_err_val /= min(itr+1, validation_every_itrs)
            print("[{}] VALIDATION: top-1 error {}, top-5 error {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                           total_t1_err_val,
                                                                           total_t5_err_val))
            # compare results
            val_error = total_t5_err_val
            total_t1_err_val, total_t5_err_val = 0, 0
            if val_error < best_val_error:
                best_val_error = val_error
                # save model
                saver.save(sess, os.path.join(logdir, "resnet50.ckpt"), g_val)
                print("[{}] Model saved to {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                      os.path.join(logdir, "resnet50.ckpt-"+str(g_val))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="resnet50", help="Which directory the model will be saved to or loaded from")
    parser.add_argument("--path_to_data", default=None, help="Specify the path to imagenet data")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of training/validation examples for each batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate to use")
    parser.add_argument("--max_itrs", type=int, default=1e5, help="Set maximum number of training iterations")
    parser.add_argument("--print_train_every_itrs", type=int, default=20, help="How often to print out training stats")
    parser.add_argument("--validation_every_itrs", type=int, default=200, help="How often to print out validation stats")
    args = parser.parse_args()

    pretrain_resnet(args.logdir,
                    args.path_to_data,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    max_itrs=args.max_itrs,
                    print_train_every_itrs=args.print_train_every_itrs,
                    validation_every_itrs=args.validation_every_itrs)
