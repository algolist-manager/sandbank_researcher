import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf
from scipy.stats import norm


def bollinger_str(n, k, data):
    port = data[['spread', 'vwap_far', 'vwap_near']].copy()

    port['ma_n'] = port['spread'].rolling(n).mean()
    port['upper'] = port['ma_n'] + k * port['spread'].rolling(n).std()
    port['lower'] = port['ma_n'] - k * port['spread'].rolling(n).std()

    port['too_high'] = port['spread'] > port['upper']
    port['high'] = (port['spread'] < port['upper']) & (port['spread'] > port['ma_n'])
    port['low'] = (port['spread'] > port['lower']) & (port['spread'] < port['ma_n'])
    port['too_low'] = port['spread'] < port['lower']
    port[['too_high', 'high', 'low', 'too_low']] = port[['too_high', 'high', 'low', 'too_low']].astype('int')

    port['long_entry'] = port['too_high'] * (port['high'] + port['low'] + port['too_low']).shift(1, fill_value=0)
    port['long_out'] = port['too_high'].shift(1, fill_value=0) * (port['low'] + port['too_low']) + port['high'].shift(1,
                                                                                                                      fill_value=0) * (
                                   port['low'] + port['too_low'])
    port['short_entry'] = port['too_low'] * (port['high'] + port['low'] + port['too_high']).shift(1, fill_value=0)
    port['short_out'] = port['too_low'].shift(1, fill_value=0) * (port['too_high'] + port['high']) + port['low'].shift(
        1, fill_value=0) * (port['too_high'] + port['high'])

    return port


def act(state, strategy):
    timing = 0

    if strategy[0] == 1:
        timing = 'long_entry'
    elif strategy[1] == 1:
        timing = 'long_out'
    elif strategy[2] == 1:
        timing = 'short_entry'
    elif strategy[3] == 1:
        timing = 'short_out'

    if strategy[4] == 0:
        if state == 'out':
            if timing == 'long_entry':
                state = 'long_in'
            elif timing == 'short_entry':
                state = 'short_in'

        elif state == 'long_in':
            if (timing == 'long_out'):
                state = 'out'

        elif state == 'short_in':
            if (timing == 'short_out'):
                state = 'out'

    else:
        if state == 'long_in':
            state = 'out'
        elif state == 'short_in':
            state = 'out'
        else:
            state = 'out'

    return state


def backtest(data, port, commission=0.00075):
    period = len(port)

    ## Timing 결정
    strategy = port[['long_entry', 'long_out', 'short_entry', 'short_out', 'rollover_out']].values
    state = 'out'
    history = [0] * (len(data))
    for i in range(len(data)):
        state = act(state, strategy[i])
        history[i] = state
    port['history'] = history

    ## Quantity 결정
    q_nt = np.zeros(period)
    q_ft = np.zeros(period)
    leverage = port['leverage'].values

    near_qt = port['vwap_near'].values
    far_qt = port['vwap_far'].values
    i = 0
    while i < len(history):
        if history[i] == 'long_in':
            entry_n = leverage[i] * 0.5 * near_qt[i]
            entry_f = -leverage[i] * 0.5 * far_qt[i]
            while history[i] == 'long_in':
                q_nt[i] = entry_n
                q_ft[i] = entry_f
                i += 1
                if i == len(history):
                    break

        elif history[i] == 'short_in':
            entry_n = -leverage[i] * 0.5 * near_qt[i]
            entry_f = leverage[i] * 0.5 * far_qt[i]
            while history[i] == 'short_in':
                q_nt[i] = entry_n
                q_ft[i] = entry_f
                i += 1
                if i == len(history):
                    break
        i += 1

    n_t = port['vwap_near'].values
    f_t = port['vwap_far'].values
    bal_t = np.zeros(period)
    pnl_t = np.zeros(period)
    com_t = np.zeros(period)

    q_t = np.array([q_nt, q_ft]).T
    ret_nt = 1 / n_t - 1 / np.roll(n_t, -1)
    ret_ft = 1 / f_t - 1 / np.roll(f_t, -1)
    ret_t = np.array([ret_nt, ret_ft])
    pnl_t[1:] = np.diag(q_t.dot(ret_t))[:-1]

    prev_q_nt = np.roll(q_nt, 1)
    prev_q_ft = np.roll(q_ft, 1)
    prev_q_nt[0] = 0
    prev_q_ft[0] = 0
    dq_nt = q_nt - prev_q_nt
    dq_ft = q_ft - prev_q_ft
    com_t = commission * (np.abs(dq_nt) / n_t + np.abs(dq_ft) / f_t)
    bal_t = pnl_t.cumsum() - com_t.cumsum()

    ret = bal_t[-1]
    std = bal_t.std()
    sharpe_ratio = bal_t[-1] / bal_t.std()
    port['bal_t'] = bal_t
    result = port

    return ret, std, sharpe_ratio, result


def rollover_out(data, period):
    rollover_ind = data[data['near_symbol'].shift(+1) != data['near_symbol']]['index'][1:] - 1
    rollover_out = np.zeros(len(data))
    for maturity in rollover_ind:
        rollover_out[maturity - period:maturity] = 1

    return rollover_out


def trade_result(port):
    i = 0
    position = np.zeros(len(port))
    history = port['history'].values
    while i < len(port):
        if history[i] == 'long_in':
            position[i] = 1
            while history[i] == 'long_in':
                i += 1
                if i >= len(port): break
                if history[i] != 'long_in':
                    position[i] = 1


        elif history[i] == 'short_in':
            position[i] = -1
            while history[i] == 'short_in':
                i += 1
                if i >= len(port): break
                if history[i] != 'short_in':
                    position[i] = -1


        else:
            i += 1

    long = np.where(position == 1)[0]
    short = np.where(position == -1)[0]

    if len(long) % 2 == 1:
        long = np.append(long, len(port) - 1)
    elif len(short) % 2 == 1:
        short = np.append(short, len(port) - 1)

    long = long.reshape((len(long) // 2, 2))
    short = short.reshape((len(short) // 2, 2))

    num_long_trade = long.shape[0]
    num_short_trade = short.shape[1]
    long_trade_returns = port['bal_t'].values[long[:, 1]] - port['bal_t'].values[long[:, 0]]
    short_trade_returns = port['bal_t'].values[short[:, 1]] - port['bal_t'].values[short[:, 0]]
    long_period = long[:, 1] - long[:, 0]
    short_period = short[:, 1] - short[:, 0]
    df = pd.DataFrame([long_trade_returns, short_trade_returns, long_period, short_period]).T
    df.columns = ['long_trade_returns', 'short_trade_returns', 'long_period', 'short_period']

    return df


def sensitive_graph(compare, n_grid, k_grid, rollover_grid):
    plt.figure(figsize=(18, 18))
    plt.subplot(3, 3, 1)
    plt.title('N ~ Ret')
    plt.xticks(n_grid)
    plt.scatter(compare.n, compare.ret)

    plt.subplot(3, 3, 2)
    plt.title('K ~ Ret')
    plt.xticks(k_grid)
    plt.scatter(compare.k, compare.ret)

    plt.subplot(3, 3, 3)
    plt.title('Rollover ~ Ret')
    plt.xticks(rollover_grid)
    plt.scatter(compare.rollover, compare.ret)

    plt.subplot(3, 3, 4)
    plt.title('N ~ Risk')
    plt.xticks(n_grid)
    plt.scatter(compare['n'], compare['std'])

    plt.subplot(3, 3, 5)
    plt.title('K ~ Risk')
    plt.xticks(k_grid)
    plt.scatter(compare['k'], compare['std'])

    plt.subplot(3, 3, 6)
    plt.title('Rollover ~ Risk')
    plt.xticks(rollover_grid)
    plt.scatter(compare['rollover'], compare['std'])

    plt.subplot(3, 3, 7)
    plt.title('N ~ Sharpe_ratio')
    plt.xticks(n_grid)
    plt.scatter(compare['n'], compare['sharpe_ratio'])

    plt.subplot(3, 3, 8)
    plt.title('K ~ Sharpe_ratio')
    plt.xticks(k_grid)
    plt.scatter(compare['k'], compare['sharpe_ratio'])

    plt.subplot(3, 3, 9)
    plt.title('Rollover ~ Sharpe_ratio')
    plt.xticks(rollover_grid)
    plt.scatter(compare['rollover'], compare['sharpe_ratio'])

    plt.show()


def metric_graph(compare):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Ret')
    plt.plot(compare['ret'].values, 'bo')
    plt.xlabel('grid')

    plt.subplot(1, 3, 2)
    plt.title('Std')
    plt.plot(compare['std'].values, 'ro')
    plt.xlabel('grid')

    plt.subplot(1, 3, 3)
    plt.title('Sharpe Ratio')
    plt.xlabel('grid')
    plt.plot(compare['sharpe_ratio'].values, 'go')

    plt.show()


def metric_summary(compare):
    print('# of grid : {0}'.format(len(compare)))
    max_ret = compare[compare['ret'] == compare['ret'].max()].values[0]
    print('Max Return : {0:.4f}, Parameter : N={1}, K={2}, Rollover={3}'
          .format(max_ret[3], max_ret[0], max_ret[1], max_ret[2]))
    min_ret = compare[compare['ret'] == compare['ret'].min()].values[0]
    print('Min Return : {0:.4f}, Parameter : N={1}, K={2}, Rollover={3}'
          .format(min_ret[3], min_ret[0], min_ret[1], min_ret[2]))
    print('\n')

    max_sr = compare[compare['sharpe_ratio'] == compare['sharpe_ratio'].max()].values[0]
    print('Max Sharpe Ratio : {0:.4f}, Parameter : N={1}, K={2}, Rollover={3}'
          .format(max_sr[5], max_sr[0], max_sr[1], max_sr[2]))
    min_sr = compare[compare['sharpe_ratio'] == compare['sharpe_ratio'].min()].values[0]
    print('Min Sharpe Ratio : {0:.4f}, Parameter : N={1}, K={2}, Rollover={3}'
          .format(min_sr[5], min_sr[0], min_sr[1], min_sr[2]))

    print('\n')

    min_std = compare[compare['std'] == compare['std'].min()].values[0]
    print('Min Standard Deviation : {0:.4f}, Parameter : N={1}, K={2}, Rollover={3}'
          .format(min_std[4], min_std[0], min_std[1], min_std[2]))
    max_std = compare[compare['std'] == compare['std'].max()].values[0]
    print('Max Standard Deviation : {0:.4f}, Parameter : N={1}, K={2}, Rollover={3}'
          .format(max_std[4], max_std[0], max_std[1], max_std[2]))


def trade_graph(trades):
    long_trade_returns = trades['long_trade_returns'].values
    short_trade_returns = trades['short_trade_returns'].values
    long_period = trades['long_period'].values
    short_period = trades['short_period'].values

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title('P&L', fontsize=20, pad=15)
    plt.scatter(x=np.arange(0, len(long_trade_returns), 1), y=long_trade_returns, label='long', c='r')
    plt.scatter(x=np.arange(0, len(short_trade_returns), 1), y=short_trade_returns, label='short', c='b')
    plt.ylim(-0.02, 0.04)
    plt.ylabel('P&L')
    plt.xlabel('trades')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Holding Period', fontsize=20, pad=15)
    plt.scatter(x=np.arange(0, len(long_period), 1), y=long_period, label='long', c='r')
    plt.scatter(x=np.arange(0, len(short_period), 1), y=short_period, label='short', c='b')
    plt.ylim(-30, 550)
    plt.ylabel('hours')
    plt.xlabel('trades')
    plt.legend()

    plt.show()


def strategy_graph(result, sample_start, sample_end):
    result['long_in'] = (result['history'] == 'long_in') * result['spread']
    result['short_in'] = (result['history'] == 'short_in') * result['spread']

    long_in = result['long_in'].values
    for i, spread in enumerate(long_in):
        if spread == 0: long_in[i] = np.nan
    result['long_in'] = long_in

    short_in = result['short_in'].values
    for i, spread in enumerate(short_in):
        if spread == 0: short_in[i] = np.nan
    result['short_in'] = short_in

    plt.figure(figsize=(15, 10))

    plt.title('Bollinger Band Strategy', fontsize=20, pad=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Spread', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.plot(result['spread'].values[sample_start:sample_end], c='black', label='spread', alpha=0.3)
    plt.plot(result['ma_n'].values[sample_start:sample_end], label='ma_n', alpha=0.3)
    plt.plot(result['upper'].values[sample_start:sample_end], label='upper', alpha=0.3)
    plt.plot(result['lower'].values[sample_start:sample_end], label='lower', alpha=0.3)
    plt.plot(result['long_in'].values[sample_start:sample_end], 'r', label='long_in')
    plt.plot(result['short_in'].values[sample_start:sample_end], 'b', label='short_in')
    plt.legend()

    plt.legend(fontsize=15)
    plt.show()


def bollinger_graph(result, sample_start, sample_end,n,k):
    result[['spread', 'ma_n', 'upper', 'lower']].iloc[sample_start:sample_end].plot(figsize=(15, 10))
    plt.title('Bollinger Band (n={0},k={1:.4f})'.format(n, k), fontsize=20, pad=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Spread', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15)
    plt.show()