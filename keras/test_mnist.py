#!/usr/bin/env python

from __future__ import print_function, division

import unittest
import mnist


class TestDataset(unittest.TestCase):
    def test_get_data(self):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist.get_data()
        self.assertEqual(x_train.shape, (50000, 28*28))
        self.assertEqual(y_train.shape, (50000, 10))
        self.assertEqual(x_val.shape, (10000, 28*28))
        self.assertEqual(y_val.shape, (10000, 10))
        self.assertEqual(x_test.shape, (10000, 28*28))
        self.assertEqual(y_test.shape, (10000, 10))
