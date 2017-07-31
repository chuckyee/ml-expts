#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
import tensorflow as tf


def generate_data(W, b, e, Npoints):
    x = np.random.rand(Npoints).astype(np.float32)
    noise = e * np.random.randn(Npoints).astype(np.float32)
    y = W * x + b + noise
    return x,y


def main():
    # parameters of data distribution
    W0 = 0.3
    b0 = 0.1
    e0 = 0.1

    # generate the data
    Ntrain, Nval, Ntest = 1000, 10, 10
    train_x, train_y = generate_data(W0, b0, e0, Ntrain)
    val_x, val_y = generate_data(W0, b0, e0, Nval)
    test_x, test_y = generate_data(W0, b0, e0, Nval)

    # hyperparameters
    learning_rate0 = 0.5
    Nepochs = 201

    # define the model
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=(None,))
        y = tf.placeholder(tf.float32, shape=(None,))
        learning_rate = tf.placeholder(tf.float32, shape=())

        W = tf.Variable(tf.random_uniform((1,), -1.0, 1.0))
        b = tf.Variable(tf.zeros((1,)))
        y_ = W * x + b
        loss = tf.reduce_mean(tf.square(y - y_))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    # execute graph
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(Nepochs):
            feed_dict = {
                x: train_x,
                y: train_y,
                learning_rate: learning_rate0
                }
            results = session.run([train, W, b, loss], feed_dict=feed_dict)
            if epoch % 20 == 0:
                print("epoch: {}  W: {}  b: {}  loss: {}".format(
                    epoch, results[1], results[2], results[3]))

if __name__ == '__main__':
    main()
