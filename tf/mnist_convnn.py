#!/usr/bin/env python

"""Convolutional neural network classifier for MNIST

Model structure:
  Conv1 = 5x5 conv (28,28, 1)->(28,28,32) -> ReLU -> 2x2 maxpool
  Conv2 = 5x5 conv (14,14,32)->(14,14,64) -> ReLU -> 2x2 maxpool
  Dense layer = (7,7,64)->1024 -> ReLU
  Dropout = 0.5 probability
  Readout = 1024-to-10 -> softmax readout

Cross-entropy loss function.

How many parameters does this model contain? Execute this snippet:

  sum([32*(5**2+1), 64*(32*5**2+1), 1024*(64*7**2+1), 10*(1024+1)])

Based on tutorial by TensorFlow authors.
"""

from __future__ import division, print_function

import argparse

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def weight_variable(shape, init_noise = 0.1, **kwargs):
    # Initialize weights with small amount of noise for symmetry breaking and
    # prevent zero gradients
    initial = tf.truncated_normal(shape, stddev = init_noise)
    return tf.Variable(initial, **kwargs)

def bias_variable(shape, bias = 0.1, **kwargs):
    # We will use ReLU neurons, init with small bias to avoid "dead neurons"
    initial = tf.constant(bias, shape = shape)
    return tf.Variable(initial, **kwargs)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x  = tf.placeholder(tf.float32, shape = [None, 784])
    y_ = tf.placeholder(tf.float32, shape = [None, 10])

    # Define convolutional net
    # First layer
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    W_conv1 = weight_variable([5, 5, 1, 32], name = 'W_conv1')
    b_conv1 = bias_variable([32], name = 'b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second layer
    W_conv2 = weight_variable([5, 5, 32, 64], name = 'W_conv2')
    b_conv2 = bias_variable([64], name = 'b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name = 'W_fc1')
    b_fc1 = bias_variable([1024], name = 'b_fc1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer
    W_fc2 = weight_variable([1024, 10], name = 'W_fc2')
    b_fc2 = bias_variable([10], name = 'b_fc2')
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Define cross entropy loss and ADAM optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(FLAGS.step_size).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add ops to save/restore all variables
    saver = tf.train.Saver()

    # Train using mini-batches
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    n_batches = int(55000 * FLAGS.epochs / FLAGS.batch_size)
    print('Number of batches = ', n_batches)
    for i in range(n_batches):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {
                x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("step {}, training accuracy {}".format(i, train_accuracy))
        train_step.run(feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    # Write trained model to file
    saver.save(sess, FLAGS.model_file)

    # Test trained model, report classification accuracy
    test_accuracy = accuracy.eval(feed_dict = {
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print("Test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'MNIST_data',
                        help = 'Directory for storing MNIST data')
    parser.add_argument('--step_size', type = float, default = 1e-4,
                        help = 'Size of ADAM optimizer step')
    parser.add_argument('--batch_size', type = int, default = 50,
                        help = 'How many training images to take for each batch')
    parser.add_argument('--epochs', type = float, default = 20.0,
                        help = 'How many times to cycle through all 55k ' \
                        'training images during training')
    parser.add_argument('--save', type = bool, default = False,
                        help = 'Whether to save trained model to file.')
    parser.add_argument('--model_file', type = str, default = 'mnist_convnn.tfl',
                        help = 'Filename for saving tensorflow model.')
    FLAGS = parser.parse_args()
    print('MNIST data directory:', FLAGS.data_dir)
    print('Gradient descent step size:', FLAGS.step_size)
    print('Batch size:', FLAGS.batch_size)
    print('Epochs:', FLAGS.epochs)
    print('Write model to disk:', FLAGS.save)
    if FLAGS.save:
        print('Filename for saving model:', FLAGS.model_file)
    tf.app.run()
