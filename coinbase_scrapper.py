import requests
import pandas as pd
from pathlib import Path
from multiprocessing import Lock
import time

h5_lock = Lock()
api_url = 'https://api.gdax.com'


def aggregate_trade_iter(symbol, last_id=1):
    last_id += 1
    while True:
        try:
            r = requests.get(f"{api_url}/products/{symbol}/trades",
                             params=dict(
                                 after=last_id + 100,
                                 before=last_id,
                             ))
            trades = r.json()
            if len(trades) == 0:
                return
            yield trades[::-1]
            last_id = trades[0]['trade_id']
        except Exception as ex:
            print(f"Error: {ex}")
            time.sleep(10)


def append_to_h5(symbol, raw_trades, path='gdax_ticks.h5'):
    df = pd.DataFrame(data=raw_trades)
    df.set_index('trade_id', inplace=True)
    h5_lock.acquire()
    with pd.HDFStore(path, 'a', complevel=4, complib='blosc:zstd') as store:
        store.append(symbol, df, format='table', min_itemsize={'price': 20, 'size': 20})
    h5_lock.release()


def handle_raw_trades(raw_trades, chunksize, symbol, store_path):
    if len(raw_trades) > chunksize:
        start = time.time()
        append_to_h5(symbol, raw_trades, path=str(store_path))
        total = time.time() - start
        print(f"Appending {len(raw_trades)} to {symbol} ticks with last idx: {raw_trades[-1]['trade_id']} "
              f"on {raw_trades[-1]['time']}. Took {total:.3f} seconds to save")
        del raw_trades[:]


def backfill_symbol(symbol, path="gdax_ticks.h5", chunksize=10000):
    store_path = Path(path)
    last_id = 1
    if store_path.exists():
        with pd.HDFStore(str(store_path), 'r') as store:
            if symbol in store:
                last_trade = store.select(symbol, start=-1)
                print(f"Resuming {symbol} from tradeid {last_trade.index[-1]}")
                last_id = last_trade.index[-1]
    raw_trades = []
    for trades in aggregate_trade_iter(symbol=product_id, last_id=last_id):
        raw_trades.extend(trades)
        time.sleep(0.1)
        handle_raw_trades(raw_trades, chunksize, symbol, store_path)
    handle_raw_trades(raw_trades, 0, symbol, store_path)


h5_store_path = "gdax_ticks.h5"
if __name__ == '__main__':

    r = requests.get(api_url + '/products')
    products = r.json()

    for product in products:
        product_id = product['id']
        raw_trades = []
        last_id = 1
        backfill_symbol(product_id, path=h5_store_path)
