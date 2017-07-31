#!/usr/bin/env python

from __future__ import print_function, division

from keras import Sequential
from keras.layers import SimpleRNN


NALPHABET = []

model = Sequential([
    SimpleRNN(27, input_dim=27, input_length=32, return_sequences=True),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(27)
])

