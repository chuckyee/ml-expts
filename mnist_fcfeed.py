#!/usr/bin/env python

"""Fully-connected 2-layer NN classifier for MNIST

Implements inference / loss / training design pattern

Based on tutorial by TensorFlow authors
"""

from __future__ import division, print_function

import time
import os.path

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

FLAGS = None


def main(_):
    """Train MNIST"""
    data_sets = input_data.read_data_sets(FLAGS.data_dir, FLAGS.fake_data)

    # Build graph: use default graph
    graph = tf.Graph()
    with graph.as_default():
        # Training input feeds
        images_placeholder = tf.placeholder(
            tf.float32, shape = (FLAGS.batch_size, mnist.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(
            tf.int32, shape = (FLAGS.batch_size,))

        # Build model: inference/loss/training + evaluation
        # Implementation in mnist.py from TensorFlow library
        logits = mnist.inference(images_placeholder,
                                 FLAGS.hidden1, FLAGS.hidden2)
        loss = mnist.loss(logits, labels_placeholder)
        train_op = mnist.training(loss, FLAGS.learning_rate)
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Reporting, initialization and checkpointing
        summary = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

    # Run session: initialize and do training loops
    with tf.Session(graph = graph) as sess:
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        # Now that everything has been built, start execution
        sess.run(init)
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            # Construct batch of MNIST images/labels to feed into NN
            batch_images, batch_labels = data_sets.train.next_batch(
                FLAGS.batch_size, FLAGS.fake_data)
            feed_dict = { images_placeholder: batch_images,
                          labels_placeholder: batch_labels }

            # Execute and fetch results: train_op is the key operation,
            # but the result we want is loss
            _, loss_value = sess.run([train_op, loss], feed_dict = feed_dict)

            duration = time.time() - start_time

            # Report training progress / write files for TensorBoard
            if step % 100 == 0:
                print('Step {}: loss = {} ({} sec)'.format(
                    step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step = step)
                


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
    flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
    flags.DEFINE_integer('batch_size', 100, 'How many training images '
                         'to take in each batch. Must divide evenly into '
                         'dataset sizes (55k for MNIST)')
    flags.DEFINE_string('data_dir', 'MNIST_data',
                        'Directory for storing MNIST data.')
    flags.DEFINE_string('train_dir', 'train_data',
                        'Directory for storing checkpoints and results.')
    flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                         'for unit testing.')
    FLAGS = flags.FLAGS
    tf.app.run()
