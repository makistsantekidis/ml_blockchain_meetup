## Machine Learning basics for Bitcoin Price Prediction

### 1. Tick Data Scraping

Download the data from coinbase using the script `coinbase_scrapper.py`. When run the script will create a `gdax.h5` file using the `pandas.HDFStore` format to save all the ticks. Because the gdax api returns the data in 100 tick batches and the total number of ticks across all pairs is 100 million++ it will take more than a week to download all the data. You may want to change the pairs to be requested to a smaller subset of pairs within the `coinbase_scrapper.py`.

You can parse the ticks downloaded into candles using the `efficient_candle_load` from the `utils.py` file.

### 2. Tweet Data Scraping

To gather tweet data you need to first setup a twitter developer account create an app and create a `secrets.json` file in the same folder as the `live_tweet_gather.py` script with your api key, secret, token and token_secret. An example of this file is as follows:

```json
{
"twitter": {
    "key": "xxxxxxxxxxxxxxxxxxxx",
    "secret": 
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "token": 
    "xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "token_secret": 
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  }
}
```

(Replace the xxxxxx with your own accounts info)

After creating the `secrets.json` file you can run `live_tweet_gather.py` to start gathering tweets with the queries specified within the script (`'bitcoin', 'BTC', 'cryptocurrency', 'crypto'` by default)

Whenever you want to prepare that data to be run with a model you need to run the `create_dataset` function within `tweets.py` . This will create a `tweets.h5` file with the tokenized tweet words.

### Run

At this point if you have gathered some data you can run `jupyter notebook` and open the `Slides.ipynb` notebook from the webui and start executing the notebook cells. Of course you can do that regardless of having data if you just want to have a nice ui to look through the code.





