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


def fill_feed_dict(data_set, images_placeholder, labels_placeholder):
    """Fills feed dictionary.

    Args:
      data_set: set of images / labels, from input_data.read_data_sets()
      images_placeholder: images placeholder
      labels_placeholder: labels placeholder

    Returns:
      feed_dict: dictionary mapping placeholder to values
    """
    batch_images, batch_labels = data_set.next_batch(
        FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_placeholder: batch_images,
        labels_placeholder: batch_labels,
    }
    return feed_dict


def evaluate(session, eval_correct, images_placeholder, labels_placeholder,
             data_set):
    """Runs one evaluation against the full epoch of data.

    Args:
      session: session in which model has been trained.
      eval_correct: Tensor which returns the number of correct predictions.
      images_placeholder: images placeholder to feed into evaluation tensor
      labels_placeholder: labels placeholder to feed into evaluation tensor
      data_set: set of images and labels to evaluate

    Returns:
      None. Writes statistics to stdout.
    """
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(
            data_set, images_placeholder, labels_placeholder)
        true_count += session.run(eval_correct, feed_dict = feed_dict)
    precision = true_count / num_examples
    fmt_str = '  Num examples: {:5}  Num correct: {:5}  Precision: {:0.04f}'
    print(fmt_str.format(num_examples, true_count, precision))


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
            feed_dict = fill_feed_dict(
                data_sets.train, images_placeholder, labels_placeholder)

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
                # Print precision against training, validation & test sets
                print('Training precision:  ', end = '')
                evaluate(sess, eval_correct, images_placeholder,
                         labels_placeholder, data_sets.train)
                print('Validation precision:  ', end = '')
                evaluate(sess, eval_correct, images_placeholder,
                         labels_placeholder, data_sets.validation)
                print('Test precision:  ', end = '')
                evaluate(sess, eval_correct, images_placeholder,
                         labels_placeholder, data_sets.test)
                

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
