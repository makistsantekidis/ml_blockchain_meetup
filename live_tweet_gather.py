from twitter import Api
import json
import dataset
import pandas as pd

tweet_field_matches = dict(
    id='tid',
    text='text',
    retweet_count='retweets',
    reply_count='replies',
    favorite_count='favorites',
    timestamp_ms='timestamp',
    user=dict(id='user_id')
)


def filter_dict(d: dict, select: dict):
    tups = []
    missing_flag = False
    for key, value in select.items():
        val = d.get(key, None)
        if val is not None:
            if type(value) is str:
                tups.append((value, val))
            elif type(value) is dict and type(val) is dict:
                sub_tup, sub_flag = filter_dict(val, value)
                tups.extend(sub_tup)
                missing_flag = (missing_flag and sub_flag)
        else:
            missing_flag = True
    return tups, missing_flag


if __name__ == '__main__':
    with open("secrets.json", 'r') as secrets_file:
        secrets = json.load(secrets_file)
    db = dataset.connect('sqlite:///tweets.db')
    tweets_table = db['tweets']
    twitter_secrets = secrets['twitter']
    CONSUMER_KEY = twitter_secrets['key']
    CONSUMER_SECRET = twitter_secrets['secret']
    ACCESS_TOKEN = twitter_secrets['token']
    ACCESS_TOKEN_SECRET = twitter_secrets['token_secret']

    api = Api(CONSUMER_KEY,
              CONSUMER_SECRET,
              ACCESS_TOKEN,
              ACCESS_TOKEN_SECRET)

    cache = []

    while True:
        try:
            test_iter = api.GetStreamFilter(track=['bitcoin', 'BTC', 'cryptocurrency', 'crypto'])
            for tweet in test_iter:
                tweet_tuples = []
                tweet_tuples, missing_flag = filter_dict(tweet, tweet_field_matches)
                tweet_dict = dict(tweet_tuples)
                if missing_flag:
                    print(tweet)
                    continue
                cache.append(tweet_dict)
                if len(cache) > 400:
                    tweets_table.insert_many(cache)
                    print(f"{pd.to_datetime('now')}: Added {len(cache)} to db")
                    del cache[:]
        except json.decoder.JSONDecodeError as ex:
            print(f"Got Decoder Exception {ex}")
        tweets_table.insert_many(cache)
        print(f"Added {len(cache)} to db")
        del cache[:]
