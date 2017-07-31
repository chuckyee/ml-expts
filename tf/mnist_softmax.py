#!/usr/bin/env python

"""A very simple MNIST classifier based on softmax.

From the TensorFlow authors.
"""

from __future__ import division, print_function

import argparse

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Define softmax model with bias: y = softmax(W.x + b)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define cross entropy loss and gradient descent optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
    #                                 reduction_indices = [1]))
    #
    # can be numerically unstable.
    #
    # Use tf.nn.softmax_cross_entropy_with_logits on raw outputs of 'y', then
    # average across the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.step_size)
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.InteractiveSession()

    # Train using mini-batches
    tf.initialize_all_variables().run()
    n_batches = int(55000 * FLAGS.epochs / FLAGS.batch_size)
    print('Number of batches = ', n_batches)
    for _ in range(n_batches):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

    # Test trained model, report classification accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_accuracy = sess.run(accuracy, feed_dict = {x : mnist.test.images,
                                                    y_: mnist.test.labels})
    print('Test set accuracy:', test_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'MNIST_data',
                        help = 'Directory for storing MNIST data')
    parser.add_argument('--step_size', type = float, default = 0.5,
                        help = 'Size of gradient descent step')
    parser.add_argument('--batch_size', type = int, default = 100,
                        help = 'How many training images to take for each batch')
    parser.add_argument('--epochs', type = float, default = 2.0,
                        help = 'How many times to cycle through all 55k ' \
                        'training images during training')
    FLAGS = parser.parse_args()
    print('MNIST data directory:', FLAGS.data_dir)
    print('Gradient descent step size:', FLAGS.step_size)
    print('Batch size:', FLAGS.batch_size)
    print('Epochs:', FLAGS.epochs)
    tf.app.run()
