import pandas as pd
import numpy as np
import numba as nb


def backtest_preds(preds, closes, slippage=0.):
    btest = Backtest(preds, closes)
    btest.run_backtest()
    pnl = btest.calculate_pnl_timeseries(slippage=slippage)
    return pnl

class Backtest:
    def __init__(self, labels, close):
        """
        Make sure labels and candles are aligned to the same indexes (datetimes) and have
        the same shape.

        :param labels:
        :param candles:
        """
        if labels.shape[0] != close.shape[0]:
            raise ValueError('Labels must have same length as candles')
        if type(labels) in [pd.DataFrame, pd.Series]:
            labels = labels.values
        if type(close) in [pd.DataFrame, pd.Series]:
            close = close.values
        self.labels = labels
        self.close = close

    def run_backtest(self, do_reverse=True, exit_on_neutral=False):
        index = np.arange(self.labels.shape[0])
        entry_points, exit_points, entry_price, exit_price, trade_type = \
            simple_backtest(self.labels, self.close, do_reverse=do_reverse, exit_on_neutral=exit_on_neutral)

        trade_type = [('long' if x else 'short') for x in trade_type]

        entry_points = index[entry_points]
        exit_points = index[exit_points]
        trades = pd.DataFrame(data=dict(entry_points=entry_points, exit_points=exit_points,
                                        entry_price=entry_price, exit_price=exit_price,
                                        trade_type=trade_type))
        trades['pnl'] = trades['entry_price'] - trades['exit_price']
        trades.loc[trades['trade_type'].isin(['long']), 'pnl'] *= -1
        self.trades = trades

    def calculate_pnl_timeseries(self, slippage=0.001, lot=None):
        """

        :param slippage: in pips
        :return:
        """
        pnls_after_slippage = (self.trades['pnl'] / self.trades['entry_price']) - slippage
        pnl_series = pd.Series(np.zeros(self.close.shape[0]))
        pnl_series.iloc[self.trades['exit_points'].values] = pnls_after_slippage.values

        if lot is not None:
            pnls_after_slippage *= lot
        return pnl_series


@nb.jit(nopython=True)
def simple_backtest(labels, closes, do_reverse=True, exit_on_neutral=False):
    long_active = False
    short_active = False
    entry_points = []
    exit_points = []
    entry_price = []
    exit_price = []
    trade_type = []

    for i in range(closes.shape[0]):

        if labels[i] == 1:  # buy
            if not long_active:
                if short_active and do_reverse:
                    exit_price.append(closes[i])
                    exit_points.append(i)
                    short_active = False

                if not short_active:
                    entry_points.append(i)
                    entry_price.append(closes[i])
                    trade_type.append(True)
                    long_active = True

        if labels[i] == 2:  # sell
            if not short_active:
                if long_active and do_reverse:
                    exit_price.append(closes[i])
                    exit_points.append(i)
                    long_active = False

                if not long_active:
                    entry_points.append(i)
                    entry_price.append(closes[i])
                    trade_type.append(False)
                    short_active = True

        if labels[i] == 0 and exit_on_neutral:
            if short_active:
                exit_price.append(closes[i])
                exit_points.append(i)
                short_active = False
            elif long_active:
                exit_price.append(closes[i])
                exit_points.append(i)
                long_active = False

    if len(entry_points) > len(exit_points):
        exit_points.append(closes.shape[0] - 1)
        exit_price.append(closes[-1])

    return entry_points, exit_points, entry_price, exit_price, trade_type


if __name__ == '__main__':
    all_pnls = []
    # Run multiple times to check backtester correctness
    for i in range(10):
        # Create random points/price to test backtester
        np.random.seed(0)
        size = 10000
        labels = np.random.choice(np.arange(3), size=size, p=[0.8, 0.1, 0.1])
        close = np.cumsum(np.random.randn(size) / 1000.) + 1.

        # Run backtester with fake labels and Close prices
        backtest = Backtest(labels, close)
        backtest.run_backtest(exit_on_neutral=False)
        pnl = backtest.calculate_pnl_timeseries(slippage=0.0000)
        all_pnls.append(pnl.cumsum().iloc[-1])
    print(np.mean(all_pnls))
