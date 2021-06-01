import os
import sys
import pickle


def read_currencies(path):
    currencies = list()
    with open(path, 'r') as file:
        for i in file.readlines():
            currencies.append(i.split()[0])
    return currencies

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(PACKAGE_DIR)
sys.path.append(ROOT_DIR)

# Paths
DATA_DIR = os.path.join(ROOT_DIR, 'data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
TIME_DATA_DIR = os.path.join(DATA_DIR, 'time_data')
if not os.path.isdir(TIME_DATA_DIR):
    os.mkdir(TIME_DATA_DIR)
TICK_DATA_DIR = os.path.join(DATA_DIR, 'tick_data')
if not os.path.isdir(TICK_DATA_DIR):
    os.mkdir(TICK_DATA_DIR)


class TimeFrame:

    def __init__(self, string, minutes):
        self.string = string
        self.minutes = minutes


# General parameters
# BASE_CURRENCY_LIST = ('BTC', 'ETH', 'LINK', 'XRP', 'BNB', 'XTZ')
# TARGET_CURRENCY_LIST = ('USDT', 'BTC')
BASE_CURRENCY_LIST = read_currencies(os.path.join(PACKAGE_DIR, 'base_currency_list.txt'))
TARGET_CURRENCY_LIST = read_currencies(os.path.join(PACKAGE_DIR, 'target_currency_list.txt'))
TIME_FRAMES = (TimeFrame('3m', 3),
               TimeFrame('15m', 15),
               TimeFrame('30m', 30),
               TimeFrame('1h', 60),
               TimeFrame('2h', 120),
               TimeFrame('4h', 240),
               TimeFrame('6h', 360))
TIME_DATA_MEMORY_IN_DAYS = 30 * 9
COLLECTOR_MEMORY = 10  # How many candle do collector save to send to the saita's inference machine


# Aggregated Trades
AGG_TRADE_COLLECTOR_MEMORY = 50
AGG_TRADE_COLLECTOR_REAL_MEMORY = 100  # Memory for calculating n_trades in a dynamic manner
TICK_BARS = (100, 500, 1000, 5000)

with open(os.path.join(PACKAGE_DIR, 'n_trades.pkl'), 'rb') as f:
    N_TRADES = pickle.load(f)
N_DERIVATIVE_AGG_TRADE = 2
GROUP_CHAT_ID = -1001119615266


# Candle Processor
PROFIT_INTERVALS = (4, 12)
N_DERIVATIVES = 2
N_NEAREST_NEIGHBORS = 100
