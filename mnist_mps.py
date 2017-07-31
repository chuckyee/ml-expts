#!/usr/bin/env python

"""Matrix product state classifier for MNIST

Based on model by Miles Stoudenmire
"""

from __future__ import absolute_import, division, print_function

import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data

FLAGS = None


def inference(images):
    init_noise = 0.1
    assert(FLAGS.feature_dim == 2) # only implement (1, x) feature map for now

    # W tensor = \prod A_i
    shape = (FLAGS.bond_dim, FLAGS.bond_dim, FLAGS.feature_dim)
    init = tf.truncated_normal(shape, stddev=init_noise)
    As = [tf.Variable(init) for i in N]

    # Phi tensor
    shape = (FLAGS.feature_dim,)
    init = tf.truncated_normal(shape, stddev=init_noise)
    Phis = [tf.Variable(init) for i in N]

    # Tensor Contraction
    tf.add(A1, tf.mul(A2, x)) for A1,A2,x in zip(As
    APhi = [tf.matmul(A, Phi) for A,Phi in zip(As, Phis)]
    # How to multiple list of tensors?

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Build graph using default graph
    graph = tf.Graph()
    with graph.as_default():
        # Training input feeds
        images_placeholder = tf.placeholder(
            tf.float32, shape = (FLAGS.batch_size, mnist.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(
            tf.float32, shape = (FLAGS.batchsize, 10)) # ten digits

        # Build model
        logits = inference(images_placeholder)

    # Run session: initialize variables and execute training iterations
    with tf.Session(graph=graph) as sess:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'MNIST_data',
                        help = 'Directory for storing MNIST data')
    FLAGS = parser.parse_args()
    print('MNIST data directory:', FLAGS.data_dir)
    tf.app.run()
