import pandas as pd
import numpy as np
import gc
from pathlib import Path
from pandas.core.base import PandasObject
from skimage.util import view_as_windows
import pickle
from tqdm import tqdm

coinbase_ticks_path = Path('gdax_ticks.h5')


def memoize():
    def decorator(original_func):
        path = Path('/Users/mike/repos/blockchain_meetup/memoized')
        path.mkdir(exist_ok=True)
        path = path / f"{original_func.__name__}.pickle"
        if not path.exists():
            cache = {}
        else:
            with open(path, 'rb') as fp:
                cache = pickle.load(fp)

        def new_func(*args, **kwargs):
            key = args + ('sentinel',) + tuple(sorted(kwargs.items()))
            if key not in list(cache.keys()):
                cache[key] = original_func(*args, **kwargs)
                with open(path, 'wb') as fp:
                    pickle.dump(cache, fp)

            return cache[key]

        return new_func

    return decorator


@memoize()
def load_targets(threshold=.025, window='1h'):
    candles = efficient_candle_load(coinbase_ticks_path, symbol='BTC-USD',
                                    start=-8000000, window=window,
                                    chunksize=1000000)
    gc.collect()
    future_price = candles.close.shift(-10).ffill().iloc[::-1]. \
                       rolling(window=5, min_periods=1).mean().iloc[::-1]
    future_direction = (future_price / candles.close) - 1

    ups = future_direction > threshold
    downs = future_direction < -threshold
    labels = pd.DataFrame(np.zeros(candles.shape[0]), index=candles.index)
    labels[ups] = 1
    labels[downs] = 2
    return labels


def roll_window(array, window_size, step_size=1):
    if isinstance(array, PandasObject):
        array = array.values
    n_dims = len(array.shape)
    if n_dims == 2:
        return view_as_windows(array,
                               window_shape=(window_size, array.shape[-1]),
                               step=step_size).copy().squeeze()
    elif n_dims == 1:
        return view_as_windows(array,
                               window_shape=window_size,
                               step=step_size).copy().squeeze()


def resample_candles(candles, window, label='right',
                     closed='right', base=0):
    return candles.resample(window, label=label, closed=closed, base=base).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }).dropna()


def efficient_candle_load(h5_path, symbol, window='1H', start=None, stop=None, chunksize=500000):
    candle_list = []
    with pd.HDFStore(h5_path, 'r') as store:
        total_rows = store.get_storer(symbol).nrows
        if start < 0:
            start = start + total_rows
        tick_iter = store.select(symbol, iterator=True, start=start,
                                 stop=stop, chunksize=chunksize)
        for i, ticks in tqdm(enumerate(tick_iter)):
            gc.collect()
            candles = parse_gdax_ticks_to_candles(ticks, window=window)
            candle_list.append(candles)
    candles = pd.concat(candle_list, axis=0)
    candles = resample_candles(candles, window=window)
    return candles


def load_candles(h5_path, symbol, window='1H', start=None, stop=None):
    df = load_ticks(h5_path, symbol, start=start, stop=stop)
    candles = parse_gdax_ticks_to_candles(df, window=window)
    return candles


def load_ticks(h5_path, symbol, start=None, stop=None):
    with pd.HDFStore(h5_path, 'r') as store:
        df = store.select(symbol, start=start, stop=stop)
    return df


def parse_gdax_ticks_to_candles(df, window='1H'):
    df.time = pd.to_datetime(df.time)
    df.price = df.price.astype(np.float32)
    df.size = df['size'].astype(np.float32)
    df = df.set_index('time')
    price_resampler = df.resample(window)['price']
    volume_resampler = df.resample(window)['size']
    candles = pd.DataFrame(data=dict(
        open=price_resampler.first(),
        high=price_resampler.max(),
        low=price_resampler.min(),
        close=price_resampler.last(),
        volume=volume_resampler.sum()
    ))
    return candles


def backtest():
    pass


if __name__ == '__main__':
    df = parse_gdax_ticks_to_candles(str(coinbase_ticks_path), 'BTC-USD', '1H')
