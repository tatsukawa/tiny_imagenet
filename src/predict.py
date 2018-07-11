import tensorflow as tf
import os
from datasets.loader import Loader
import models.vgg16
import numpy as np
from tensorflow.contrib.slim.nets import vgg
import csv
slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)


def clip(x):
    EPS = 1e-10
    return tf.clip_by_value(x, EPS, 1.0-EPS)


def cross_entropy(x, y):
    return -tf.reduce_sum(y * tf.log(clip(x)), reduction_indices=[1])


loader = Loader()

with tf.Graph().as_default():
    data_op, init_op, names = loader.get_test_data()
    X_op, Y_op = data_op

    image_p = tf.placeholder(tf.float32, [None, loader.H, loader.W, loader.CHANNELS])
    label_p = tf.placeholder(tf.float32, [None, len(loader.raw_labels)])
    inference_op = models.vgg16.inference(image_p)
    logit_op = tf.nn.softmax(inference_op)
    output_label_op = tf.argmax(logit_op, axis=1)

    saver = tf.train.Saver(max_to_keep=10000)

    ckpt_path = '../ckpt/batch_size-500/model.ckpt-1800'

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    )

    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), init_op])
        saver.restore(sess, ckpt_path)
        test_x = sess.run([X_op])
        labels = sess.run(output_label_op, feed_dict={image_p: test_x[0]})
        labels = [loader.convert_num_to_label[label] for label in labels]

        with open('output.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n', delimiter="\t")
            for name, label, in zip(names, labels):
                writer.writerow([name, label])

        with open('output_including_num.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n', delimiter="\t")
            for name, label, in zip(names, labels):
                writer.writerow([name, label, loader.convert_label_to_num[label]])

