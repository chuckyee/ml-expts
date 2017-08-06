#!/usr/bin/env python

import numpy as np

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


def create_model(vocabulary_size=5000,
                 embedding_dim=32,
                 max_review_length=500,
                 hidden_dim=100):
    layers = [
        Embedding(vocabulary_size, embedding_dim, input_length=max_review_length),
        LSTM(hidden_dim),
        Dense(1, activation='sigmoid'),
    ]
    model = Sequential(layers)

    return model

def load_data(max_review_length, vocabulary_size=None):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=vocabulary_size)

    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

    return (x_train, y_train), (x_test, y_test)

def train():
    vocabulary_size = 5000      # number of most frequent words to keep
    embedding_dim = 32          # dim of vector space to embed vocabulary words
    max_review_length = 500     # truncate reviews to this length
    hidden_dim = 100

    epochs = 3
    batch_size = 64
    

    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data(
        max_review_length, vocabulary_size)

    print("Building model...")
    model = create_model(
        vocabulary_size=vocabulary_size, embedding_dim=embedding_dim,
        max_review_length=max_review_length, hidden_dim=hidden_dim)

    print("Building loss...")
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    
    print("Training model...")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(scores)

if __name__ == '__main__':
    train()
