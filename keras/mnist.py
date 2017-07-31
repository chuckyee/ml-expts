#!/usr/bin/env python

from __future__ import print_function, division

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras import utils

def get_data():
    MAX_VALUE = 255
    NUM_CLASSES = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # split out 10000 of training set into validation set
    x_val = x_train[50000:,:,:].reshape(10000, 784) / MAX_VALUE
    y_val = utils.to_categorical(y_train[50000:], NUM_CLASSES)

    # new training set has 50000 digits
    x_train = x_train[:50000,:,:].reshape(50000, 784) / MAX_VALUE
    y_train = utils.to_categorical(y_train[:50000], NUM_CLASSES)

    x_test = x_test.reshape(10000, 784) / MAX_VALUE
    y_test = utils.to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def main(args):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data()
    
    model = Sequential([
        Dense(args.units, activation='relu', input_dim=784),
        Dropout(args.dropout),
        Dense(args.units, activation='relu'),
        Dropout(args.dropout),
        Dense(10, activation='softmax'),
    ])

    model.summary()

    optimizers = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    if args.optimizer not in optimizers:
        raise Exception("Unknown optimizer ({}).".format(args.optimizer))

    optimizer_args = {} if (args.learning_rate is None) else {'lr': args.learning_rate}
    optimizer = optimizers[args.optimizer](**optimizer_args)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=(x_val, y_val))

    loss, accuracy = model.evaluate(x_test, y_test)

    print()
    print("Test loss: ", loss)
    print("Test accuracy: ", accuracy)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Simple densely-connect network for MNIST.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', default=20, type=int, help='Number of training epochs.')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='Mini-batch size during training.')
    parser.add_argument('-d', '--dropout', default=0.2, type=float, help='Fraction of inputs to randomly drop.')
    parser.add_argument('-u', '--units', default=512, type=int, help='Number of units in hidden layers.')
    parser.add_argument('-o', '--optimizer', default='rmsprop', type=str,
                        help='Optimizer: sgd, rmsprop, adagrad, adadelta, adam, adamax, or nadam.')
    parser.add_argument('-r', '--learning_rate', type=float, help='Optimizer learning rate.')
    args = parser.parse_args()
    main(args)
