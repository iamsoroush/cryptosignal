import os
import sys
import logging
from datetime import datetime
from time import sleep
import pytz

import pandas as pd

from . import ROOT_DIR


kline_mapper = {'kline_start_time': 't',
                'kline_close_time': 'T',
                'time_frame': 'i',
                'pair': 's',
                'first_trade_id': 'f',
                'last_trade_id': 'L',
                'Open': 'o',
                'Close': 'c',
                'High': 'h',
                'Low': 'l',
                'Volume': 'v',  # base asset volume, use 'q' for target asset's volume
                'number_of_trades': 'n',
                'is_closed': 'x'}


agg_trade_mapper = {'event_time': 'E',  # int
                    'pair': 's',  # str
                    'agg_trade_id': 'a',  # int
                    'price': 'p',  # str
                    'volume': 'q',  # str
                    'trade_time': 'T',  # int
                    'buyer_maker': 'm',  # bool
                    }


def try_decorator(func):

    def wrapper(*args, **kwargs):
        for i in range(max_retries + 1):
            try:
                ret = func(*args, **kwargs)
            except Exception as e:
                if i < max_retries:
                    sleep(sleep_seconds)
                else:
                    raise e
            else:
                return ret

    sleep_seconds = 2
    max_retries = 5

    return wrapper


def read_binance_api():
    api_key = None
    api_secret = None
    with open(os.path.join(ROOT_DIR, '../binance.txt'), 'r') as file:
        for line in file.readlines():
            if line.startswith('API'):
                api_key = line.split(' ')[-1]
            elif line.startswith('secret'):
                api_secret = line.split(' ')[-1]
    return api_key, api_secret


def get_logger(name, write_logs=True):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if write_logs:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(dir_path, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, "{}.log".format(name)))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


def save_df(df, save_path):
    df.to_csv(save_path)


def load_df(path):
    return pd.read_csv(path, index_col='Timestamp')


def str_to_milisecond_timestamp(str_datetime):

    """:arg str_datetimea: '%Y-%m-%d %H:%M:%S.%f', e.g. 2019-06-06 00:00:00.0"""

    dt = datetime.strptime(str_datetime, '%Y-%m-%d %H:%M:%S.%f')
    dt_miliseconds = int(datetime.timestamp(dt)) * 1000
    return dt_miliseconds


def miliseconds_timestamp_to_str(ms_timestamp):

    """:returns str datetime in %Y-%m-%d %H:%M:%S.%f format."""

    return datetime.fromtimestamp(ms_timestamp / 1000).isoformat()


def tehran_msts_to_str(ts):

    """Miliseconds-timestamp to Tehran str datetime."""

    dt = datetime.fromtimestamp(ts / 1000)  # Assuming that the bot is running on the server with UTC timezone
    dt = _timezone_converter(dt, current_tz='UTC', target_tz='Asia/Tehran')
    return dt.strftime("%Y-%m-%d %H:%M:%S")
    # return dt.astimezone(pytz.timezone('Asia/Tehran')).timestamp()


def _timezone_converter(input_dt, current_tz, target_tz):
    current_tz = pytz.timezone(current_tz)
    target_tz = pytz.timezone(target_tz)
    target_dt = current_tz.localize(input_dt).astimezone(target_tz)
    return target_tz.normalize(target_dt)

