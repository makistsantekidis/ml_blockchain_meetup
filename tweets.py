from pathlib import Path
import dataset
import pandas as pd
import numpy as np
import spacy
from spacy_cld import LanguageDetector

from tqdm import tqdm
from multiprocessing import Pool
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import gc
from utils import memoize

db_path = Path('/Users/mike/repos/social-trader/scratch/tweets.db')


def create_dataset():
    db = dataset.connect(f"sqlite:///{str(db_path)}")
    nlp = spacy.load('en')
    language_detector = LanguageDetector()
    nlp.add_pipe(language_detector)
    # doc = nlp('This is some English text.')
    # doc._.languages

    tweets_table = db['tweets']
    dataset_samples = []
    ids = set()
    tokenizer = Tokenizer(num_words=10000)
    # if Path('tweets2.h5').exists():
    #     with pd.HDFStore('tweets.h5', 'r') as store:
    #         store.select('tweets', columns=['tid'])
    texts = []
    timestamp_list = []
    tweet_list = []
    for i, tweet in tqdm(enumerate(tweets_table)):
        # if i > 10000:
        #     break
        if tweet['tid'] in ids:
            continue
        else:
            ids.add(tweet['tid'])
        text = tweet['text']
        # doc = nlp(text)
        # if doc._.languages and doc._.languages[0] != 'en':
        #     continue
        # text = ' '.join([token.lemma_ for token in doc])

        tweet_list.append((tweet['timestamp'], text, tweet['tid']))
    df = pd.DataFrame(data=tweet_list, columns=['timestamp', 'text', 'tid'])
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    tokenizer.fit_on_texts(df['text'].values)
    with open('tokenizer', 'wb') as fp:
        pickle.dump(tokenizer, fp)
    seqs = tokenizer.texts_to_sequences(df['text'].values)
    df.drop('text', axis=1, inplace=True)
    seqs = np.asarray(pad_sequences(seqs, maxlen=100))
    df = pd.concat([df, pd.DataFrame(seqs)], axis=1)
    df.set_index('timestamp', inplace=True)
    with pd.HDFStore('tweets.h5', 'a', complevel=6, complib='blosc:zstd') as store:
        store.append('tweets', df, format='t')


@memoize()
def load_dataset(window='1h', block_size=100, chunksize=200000):
    tweet_h5_path = Path('/Users/mike/repos/blockchain_meetup/tweets.h5')
    concated_tweets = []
    with pd.HDFStore(str(tweet_h5_path), 'r') as store:
        # tweets = store.select('tweets', start=-10000000)
        tweets_iter = store.select('tweets', chunksize=chunksize)
        last_tweets = None
        for tweets in tweets_iter:
            gc.collect()
            if last_tweets is None:
                idx_cur = tweets.index[0].ceil(window)
            else:
                tweets = pd.concat([last_tweets, tweets], axis=0)
            idx_end = tweets.index[-1].ceil(window)
            while True:
                concated_tweets.append(
                    np.concatenate([[pd.to_datetime(idx_cur)],
                                    tweets.loc[tweets.index < idx_cur].iloc[-block_size:, 1:].values.flatten()]))
                idx_cur += pd.to_timedelta(window)
                if idx_cur + pd.to_timedelta(window) >= idx_end:
                    break
            last_tweets = tweets.iloc[-chunksize:].copy()

    # idx_cur = tweets.index[0].ceil(window)
    # idx_end = tweets.index[-1].ceil(window)
    # idx_cur += pd.to_timedelta(window)
    # while True:
    #     concated_tweets.append(
    #         [idx_cur] +
    #         tweets.loc[tweets.index < idx_cur].iloc[-block_size:, 1:].values.flatten().tolist())
    #     idx_cur += pd.to_timedelta(window)
    #     if idx_cur + pd.to_timedelta(window) > idx_end:
    #         break
    concated_tweets = pd.DataFrame(concated_tweets)
    concated_tweets.set_index(0, inplace=True)
    return concated_tweets


if __name__ == '__main__':

    gc.collect()
    create_dataset()
