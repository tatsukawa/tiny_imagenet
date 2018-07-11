import tensorflow as tf
import os
from datasets.loader import Loader
import models.vgg16
import numpy as np
from tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)


def optimizer(loss):
    """Add training operation, global_step and learning rate variable to Graph
    Args:
      loss: model loss tensor
      config: training configuration object
    Returns:
      (train_op, global_step, lr)
    """
    lr = tf.Variable(0.01, trainable=False, dtype=tf.float32)

    tf.summary.scalar('Learning rate', lr)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    train_op = opt.minimize(loss, global_step=global_step)
    grads = opt.compute_gradients(loss)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    return train_op, global_step, lr


def clip(x):
    EPS = 1e-10
    return tf.clip_by_value(x, EPS, 1.0-EPS)


def cross_entropy(x, y):
    return -tf.reduce_sum(y * tf.log(clip(x)), reduction_indices=[1])


def loss(logits, labels):
    ce = cross_entropy(logits, labels)#tf.reduce_mean(cross_entropy(logits, labels))
    cross_entropy_mean = tf.reduce_mean(ce, name='cross_entropy')
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularizer = slim.l2_regularizer(0.0005)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

    tf.summary.scalar('L2 Regularization term', reg_term)

    return cross_entropy_mean + reg_term


def accuracy(preds, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1)), tf.float32))


def count_true_prediction(preds, labels):
    return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1)), tf.float32))


loader = Loader()
MAX_EPOCH = 10000
batch_size = 500
MAX_STEP = int(500*200 / batch_size)


with tf.Graph().as_default():
    data_op, init_op = loader.get_train_batch(is_augment=True, batch_size=batch_size)
    X_op, Y_op = data_op

    val_data_op, init_val_op = loader.get_validation_data()
    val_X_op, val_Y_op = val_data_op

    image_p = tf.placeholder(tf.float32, [None, loader.H, loader.W, loader.CHANNELS])
    label_p = tf.placeholder(tf.float32, [None, len(loader.raw_labels)])
#    inference_op = models.cnn.vgg_16(image_p)
    inference_op = models.vgg16.inference(image_p)
    logit_op = tf.nn.softmax(inference_op)
    loss_op = loss(logit_op, label_p)
    train_op = optimizer(loss_op)
    accuracy_op = accuracy(logit_op, label_p)

    summary_image_op = tf.summary.image('Input image', image_p, 10)

    saver = tf.train.Saver(max_to_keep=10000)

    import datetime
    date = datetime.datetime.now()
    id = '{}-{}-{}_{}_{}_{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)

    ckpt_path = '../ckpt/{}'.format(id)
    ckpt = tf.train.get_checkpoint_state(ckpt_path)

    logs_dir = '../ckpt/{}'.format(id) #TODO:

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    )

    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), init_op, init_val_op])

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

        if ckpt is None:
            # ref: https://www.kaggle.com/ardiya/tensorflow-vgg-pretrained
            restore = slim.assign_from_checkpoint_fn(
                '../pretrained/vgg_16.ckpt',
                slim.get_model_variables("vgg_16"))
            restore(sess)
        else:
            saver.restore(sess, ckpt.model_checkpoint_path)

        saver.save(sess, os.path.join(ckpt_path, 'model.ckpt'), global_step=0)

        for epoch in range(MAX_EPOCH):
            for step in range(MAX_STEP):
                x, y = sess.run([X_op, Y_op])

                summary = sess.run(summary_op, feed_dict={image_p: x, label_p: y})
                summary_writer.add_summary(summary, global_step=epoch*MAX_STEP+step)

                sess.run(train_op, feed_dict={image_p: x, label_p: y})

                logit = sess.run(logit_op, feed_dict={image_p: x})

                loss = sess.run(loss_op, feed_dict={image_p: x, label_p: y})
                acc = sess.run(accuracy_op, feed_dict={image_p: x, label_p: y})

                summary = tf.Summary()

                print('step: {: ^5}, loss = {:.6f}, train accuracy = {:.6f}'.format(
                   step, loss, acc,
                ))

                summary.value.add(tag='Train batch loss', simple_value=loss)
                summary_writer.add_summary(sess.run(summary_image_op, feed_dict={image_p: x}), global_step=epoch*MAX_STEP+step)
                summary_writer.add_summary(summary, global_step=epoch*MAX_STEP+step)

            summary = tf.Summary()
            val_acc = 0
            val_loss = 0
            for i in range(20):
                val_x, val_y = sess.run([val_X_op, val_Y_op])
                val_acc += sess.run(accuracy_op, feed_dict={image_p: val_x, label_p: val_y})
                val_loss += sess.run(loss_op, feed_dict={image_p: val_x, label_p: val_y})
            val_acc /= 20.0
            val_loss /= 20.0
            summary.value.add(tag='Validation loss', simple_value=val_loss)
            summary.value.add(tag='Validation accuracy', simple_value=val_acc)
            summary.value.add(tag='Train batch loss', simple_value=loss)
            summary_writer.add_summary(summary, global_step=epoch)

            print('epoch: {:}, val accuracy = {:.6f}, val loss = {:.6f}'.format(
                epoch, val_acc, val_loss
            ))

            saver.save(sess, os.path.join(ckpt_path, 'model.ckpt'), global_step=epoch*MAX_STEP)

        summary_writer.close()

