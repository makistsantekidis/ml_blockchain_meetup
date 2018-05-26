from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, PReLU
from keras.layers import Masking, GlobalAveragePooling1D, Embedding, Dense
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical

import pickle, gc

from tweets import load_dataset
from utils import load_targets, efficient_candle_load
import numpy as np
import pandas as pd
from pathlib import Path


def load_train_test(train_prc=0.7, window='10min'):
    tweets = load_dataset(window=window)
    gc.collect()
    targets = load_targets(window=window, threshold=.004)
    gc.collect()
    tweets.index = pd.to_datetime(tweets.index)
    tweets, targets = tweets.align(targets, join='inner', axis=0)
    # tweets = np.asarray([i[0] for i in tweets.values])
    indexes = targets.index
    targets = to_categorical(targets.values, num_classes=3)
    num_samples = indexes.shape[0]
    num_train = int(num_samples * train_prc)
    train_features = tweets[:num_train]
    test_features = tweets[num_train:]
    train_labels = targets[:num_train]
    test_labels = targets[num_train:]
    train_idx = indexes[:num_train]
    test_idx = indexes[num_train:]
    return train_features, train_labels, train_idx, test_features, test_labels, test_idx


if __name__ == '__main__':
    tokenizer = None
    num_words = 10000
    if Path('tokenizer').exists():
        with open('tokenizer', 'rb') as fp:
            tokenizer = pickle.load(fp)
            num_words = tokenizer.num_words
    del tokenizer
    gc.collect()
    train_features, train_labels, train_idx, test_features, test_labels, test_idx = load_train_test(window='30min')
    print(np.unique(train_labels.argmax(axis=-1), return_counts=True))
    print(np.unique(test_labels.argmax(axis=-1), return_counts=True))
    model = Sequential()
    model.add(Embedding(10000, 200, input_length=10000))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64))
    model.add(PReLU())
    model.add(Dense(3, activation='softmax'))
    optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy',
                  # optimizer=Adam(lr=0.001),
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(train_features, train_labels,
              batch_size=16,
              epochs=100,
              validation_data=(test_features, test_labels), verbose=2)
    train_preds = model.predict(train_features)
    test_preds = model.predict(test_features)
    print(np.unique(train_preds.argmax(axis=-1), return_counts=True))
    print(np.unique(test_preds.argmax(axis=-1), return_counts=True))
