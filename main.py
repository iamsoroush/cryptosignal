import itertools
import datetime
import threading
from time import sleep
from requests.exceptions import ConnectTimeout
import signal
import sys

from binance.client import Client
from binance.websockets import BinanceSocketManager
from ccxt.base.errors import RequestTimeout

from src import TIME_FRAMES,\
    BASE_CURRENCY_LIST,\
    TARGET_CURRENCY_LIST,\
    TIME_DATA_MEMORY_IN_DAYS,\
    TIME_DATA_DIR
from src.iohandler import sub_candle_collector, read_binance_api, mapper
from src.saita import SAITA
from src.data_handling import TimeDataHandler
from src.utils import get_logger, miliseconds_timestamp_to_str
from src.saita_bot import SAITABot
from src.database import DBHandler


# Create logger
logger = get_logger(name=__name__, write_logs=True)

# Connect to database
db_handler = DBHandler()
db_handler.start()


def signal_handler(sig, frame):
    manager.close()
    print('closed binance websocket manager.')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_saita():
    while True:
        try:
            saita = SAITA()
        except:
            sleep(2)
            continue
        else:
            break
    return saita


def _get_saita_bot():
    # Create telegram bot
    token = '1069900023:AAGU8F0vdcAYxewlhbzsK8hxmfkggqqkbgs'
    saita_bot = SAITABot(token, db_handler)
    return saita_bot


def candle_callback(msg):
    splitted = msg['stream'].split('@')
    pair = splitted[0].upper()
    time_frame = splitted[1].split('_')[1]

    candle_data = msg['data']['k']
    is_closed = candle_data[mapper['is_closed']]
    if is_closed:
        str_dt = miliseconds_timestamp_to_str(int(candle_data[mapper['kline_close_time']]))
        logger.info('received a closed candle for {}/{}'.format(pair, time_frame))
        c = {'Open': float(candle_data[mapper['Open']]),
             'Close': float(candle_data[mapper['Close']]),
             'High': float(candle_data[mapper['High']]),
             'Low': float(candle_data[mapper['Low']]),
             'Volume': float(candle_data[mapper['Volume']]),
             'DateTime': str_dt}
        base_candle_collectors[pair].send(c)


def _create_binance_websocket_manager(pairs, base_time_frame):
    api_key, api_secret = read_binance_api()
    client = Client(api_key=api_key, api_secret=api_secret)
    manager = BinanceSocketManager(client, user_timeout=60)

    streams = [pair.lower() + '@kline_' + base_time_frame for pair in pairs]
    conn_key = manager.start_multiplex_socket(streams,
                                              candle_callback)
    logger.info('Connection keys: {}'.format(conn_key))
    return manager


def start_binance_websocket_manager():
    # Start binance data fetcher
    while True:
        try:
            manager = _create_binance_websocket_manager(valid_pairs, TIME_FRAMES[0].string)
            manager.start()
        except ConnectTimeout:
            sleep(2)
            continue
        else:
            break
    logger.info('binance websocket started listening ...')
    return manager


def start_telegram_bot():
    saita_bot = _get_saita_bot()
    bot = saita_bot.updater.bot
    while True:
        try:
            saita_bot.run()
        except:
            sleep(2)
            continue
        else:
            break
    return bot


def get_data_handler():
    while True:
        try:
            data_handler = TimeDataHandler()
        except RequestTimeout:
            sleep(2)
            continue
        else:
            break
    return data_handler


def collect_data(data_handler):

    """Download new time-data from ccxt each saturday at 23:59:59"""

    now = datetime.datetime.now()
    if now.weekday() != 5:  # saturday
        logger.info('today is {}th day of the week. skipping the data downloading.'.format(now.weekday()))
        return
    else:
        for base_currency, target_currency in itertools.product(BASE_CURRENCY_LIST, TARGET_CURRENCY_LIST):
            data_handler.update_time_data(base_currency,
                                          target_currency,
                                          TIME_DATA_MEMORY_IN_DAYS)


if __name__ == '__main__':
    # Initiate a HistoricalDataHandler object
    data_handler = get_data_handler()

    # Update time-data
    for base_currency, target_currency in itertools.product(BASE_CURRENCY_LIST, TARGET_CURRENCY_LIST):
        data_handler.update_time_data(base_currency,
                                      target_currency)

    # Create SAITA's processing unit, SAITA-core
    saita = get_saita()

    # Start telegram bot
    bot = start_telegram_bot()

    # Create candle-collectors
    base_time_frame = TIME_FRAMES[0]
    valid_pairs = [i[0] + i[1] for i in list(itertools.product(BASE_CURRENCY_LIST, TARGET_CURRENCY_LIST))]
    base_candle_collectors = dict()
    for pair in valid_pairs:
        print(pair, ':')
        sub_collectors = list()
        for time_frame in TIME_FRAMES[1:]:
            collector = sub_candle_collector(time_frame, base_time_frame, pair, saita, bot, db_handler)
            next(collector)
            sub_collectors.append(collector)

        base_collector = sub_candle_collector(base_time_frame,
                                              base_time_frame,
                                              pair,
                                              saita,
                                              bot,
                                              db_handler,
                                              sub_collectors)
        next(base_collector)
        base_candle_collectors[pair] = base_collector

    # Start binance's websocket
    manager = start_binance_websocket_manager()

    # Start a loop to collect
    while True:
        now = datetime.datetime.now()
        next_call_dt = now + datetime.timedelta(hours=23 - now.hour,
                                                minutes=59 - now.minute,
                                                seconds=59 - now.second)
        seconds_to_next_midnight = float((next_call_dt - now).seconds)
        timer = threading.Timer(seconds_to_next_midnight,
                                collect_data,
                                args=(data_handler,),
                                daemon=True)
        timer.start()
        timer.join()
