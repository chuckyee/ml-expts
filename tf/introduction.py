#!/usr/bin/env python

from __future__ import division, print_function

import tensorflow as tf
import numpy as np

# Create 100 phony x,y data points in NumPy, y = 0.1 * x + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1 * x_data + 0.3

# Find values for W and b and approximate y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initilize variables
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the line
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# Should learn best fit is W: [0.1], b: [0.3]
