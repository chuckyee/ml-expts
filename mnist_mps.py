#!/usr/bin/env python

"""Matrix product state classifier for MNIST

Based on model by Miles Stoudenmire
"""

from __future__ import division, print_function

import argparse

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x  = tf.placeholder(tf.float32, shape = [None, 784])
    y_ = tf.placeholder(tf.float32, shape = [None, 10])

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'MNIST_data',
                        help = 'Directory for storing MNIST data')
    FLAGS = parser.parse_args()
    print('MNIST data directory:', FLAGS.data_dir)
    tf.app.run()
