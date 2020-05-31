import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_dir)

# Paths
DATA_DIR = os.path.join(project_dir, 'data')
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
BASE_CURRENCY_LIST = ('BTC', 'ETH', 'LINK', 'XRP', 'BNB', 'XTZ')
TARGET_CURRENCY_LIST = ('USDT',)
TIME_FRAMES = (TimeFrame('3m', 3),
               TimeFrame('15m', 15),
               TimeFrame('30m', 30),
               TimeFrame('1h', 60),
               TimeFrame('2h', 120),
               TimeFrame('4h', 240),
               TimeFrame('6h', 360))
# TIME_FRAMES = ('3m', '15m', '30m', '1h', '2h', '4h', '6h')
TICK_BARS = (100, 500, 1000, 2000, 5000, 10000)
TIME_DATA_MEMORY_IN_DAYS = 30 * 9

# Candle Processor
PROFIT_INTERVALS = (4, 12)
N_DERIVATIVES = 3
N_NEAREST_NEIGHBORS = 100
COLLECTOR_MEMORY = 10  # How many candle do collector save to send to the saita's inference machine
