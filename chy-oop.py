#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self):
        pass

    def build(self):
        x = tf.placeholder(tf.float32, shape=(None,))
        y = tf.placeholder(tf.float32, shape=(None,))
        learning_rate = tf.placeholder(tf.float32, shape=())

        W = tf.Variable(tf.random_normal(shape=(1,)))
        b = tf.Variable(tf.zeros(shape=(1,)))
        y_ = W*x + b

        loss = tf.reduce_mean(tf.square(y - y_))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)
            
    def train(self, session):
        results = session.run([self.train], feed_dict=feed_dict)

    def validate(self, session):
        session.run([], feed_dict=feed_dict)
        pass

    @property
    def parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            dims = [dim.value for dim in variable.get_shape()]
            variable_parameters = 1
            for dim in dims:
                variable_parameters *= dim
            total_parameters += variable_parameters
        return total_parameters

def main():
    model = Model()

    print("Building model...", end="")
    graph = tf.Graph()
    with graph.as_default():
        model.build()
    print("done.")

    with tf.Session(graph=graph) as session:
        print("Number of parameters: {}".format(model.parameters))

        print("Training model...")
        for epoch in range(Nepochs):
            results = model.train(x_train, y_train)
            if epoch % 20 == 0:
                results = model.test(x_val, y_val)

    # better interface design
    model.build()

    with model.session:
        print(model.parameters)
        for epoch in range(Nepochs):
            model.train()


if __name__ == "__main__":
    main()
