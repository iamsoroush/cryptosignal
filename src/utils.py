import os
import sys
import logging
from datetime import datetime

import pandas as pd


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

