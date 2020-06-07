import itertools
import datetime
import threading
import signal
import sys
import pickle
import argparse

import numpy as np
from binance.client import Client
from binance.websockets import BinanceSocketManager

from src import TIME_FRAMES,\
    BASE_CURRENCY_LIST,\
    TARGET_CURRENCY_LIST,\
    N_TRADES
from src.iohandler import sub_candle_collector, agg_trade_coroutine
from src.saita import SAITA
from src.saita_bot import SAITABot
from src.data_handling import TimeDataHandler, TickDataHandler
from src.utils import get_logger, read_binance_api, kline_mapper, agg_trade_mapper, try_decorator, get_tehran_ts
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


def candle_callback(msg):
    if 'e' in list(msg.keys()):
        logger.error('Binance websocket manager: '.format(msg['m']))
    else:
        data = msg['data']
        pair = data['s']
        if data['e'] == 'kline':
            candle_data = msg['data']['k']
            time_frame = candle_data[kline_mapper['time_frame']]
            is_closed = candle_data[kline_mapper['is_closed']]
            if is_closed:
                tehran_ts = float(get_tehran_ts(candle_data[kline_mapper['kline_close_time']]))
                logger.info('received a closed candle for {}/{}'.format(pair, time_frame))
                c = {'Open': float(candle_data[kline_mapper['Open']]),
                     'Close': float(candle_data[kline_mapper['Close']]),
                     'High': float(candle_data[kline_mapper['High']]),
                     'Low': float(candle_data[kline_mapper['Low']]),
                     'Volume': float(candle_data[kline_mapper['Volume']]),
                     'DateTime': tehran_ts}
                kline_base_candle_collectors[pair].send(c)
        else:
            trade_tehran_ts = float(get_tehran_ts(data[agg_trade_mapper['trade_time']]))
            price = float(data[agg_trade_mapper['price']])
            volume = float(data[agg_trade_mapper['volume']])
            trade = [trade_tehran_ts, price, volume]
            agg_trade_candle_collectors[pair].send(trade)


def _get_usdt_pairs(client):
    info = client.get_exchange_info()
    return [s['symbol'] for s in info['symbols'] if s['symbol'].endswith('USDT')
            and not any(elem in s['symbol'].split('USDT')[0] for elem in ['USD', 'BTC', 'ETH', 'BNB', 'XRP'])]


def _add_sockets(manager, pairs, base_time_frame, all_usdt_pairs, agg_trade_streams):
    streams = list()
    if agg_trade_streams:
        agg_trade_streams = [pair.lower() + '@aggTrade' for pair in all_usdt_pairs]
        streams.extend(agg_trade_streams)
    kline_streams = [pair.lower() + '@kline_' + base_time_frame for pair in pairs]
    streams.extend(kline_streams)

    conn_key = manager.start_multiplex_socket(streams,
                                              candle_callback)
    logger.info('Connection keys: {}'.format(conn_key))


@try_decorator
def start_binance_websocket_manager(agg_trade_streams=False):
    api_key, api_secret = read_binance_api()
    client = Client(api_key=api_key, api_secret=api_secret)
    manager = BinanceSocketManager(client, user_timeout=60)

    # For time-based candles
    base_time_frame = TIME_FRAMES[0]
    valid_pairs = [i[0] + i[1] for i in list(itertools.product(BASE_CURRENCY_LIST, TARGET_CURRENCY_LIST))]

    # # For aggregated-trade candles, except BTC, ETH, BNB, XRP
    # all_usdt_pairs = _get_usdt_pairs(client)
    # n_trades = _get_n_trades_for_alts(all_usdt_pairs)

    # Add sockets to manager
    _add_sockets(manager, valid_pairs, base_time_frame.string, list(N_TRADES.keys()), agg_trade_streams)

    # Generate time-based collectors for base time-frame
    global kline_base_candle_collectors
    kline_base_candle_collectors = generate_kline_base_candle_collectors(valid_pairs, base_time_frame)

    # Generate agg-trade collectors
    if agg_trade_streams:
        global agg_trade_candle_collectors
        agg_trade_candle_collectors = generate_agg_trade_base_candle_collectors(list(N_TRADES.keys()))

    manager.start()
    logger.info('binance websocket started listening ...')
    return manager


@try_decorator
def get_saita():
    saita = SAITA()
    return saita


def get_saita_bot(db_handler):
    # Create telegram bot
    token = '1069900023:AAGU8F0vdcAYxewlhbzsK8hxmfkggqqkbgs'
    saita_bot = SAITABot(token, db_handler)
    return saita_bot


@try_decorator
def start_telegram_bot(db_handler):
    saita_bot = get_saita_bot(db_handler)
    bot = saita_bot.updater.bot
    saita_bot.run()
    return bot


@try_decorator
def get_data_handler():
    data_handler = TimeDataHandler()
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
                                          target_currency)


def generate_kline_base_candle_collectors(valid_pairs, base_time_frame):
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
    return base_candle_collectors


def generate_agg_trade_base_candle_collectors(pairs):
    candle_collectors = dict()
    for pair in pairs:
        n_trades = N_TRADES[pair]
        collector = agg_trade_coroutine(pair, 1, n_trades, saita, bot, db_handler)
        next(collector)
        candle_collectors[pair] = collector
    return candle_collectors


def _get_n_trades_for_alts(pairs, quantile=15):
    n_trades = dict()
    handler = TickDataHandler()
    for pair in pairs:
        # logger.info('fetching the agg_trade data for {} for last 7 days.'.format(pair))
        agg_trades = handler._fetch_last_2weeks_data(pair)
        if len(agg_trades) < 100:
            logger.info('skipping')
            continue
        diffs = (agg_trades['Timestamp'].iloc[10::5].values - agg_trades['Timestamp'].iloc[:-10:5].values) / 1000

        # in 90% of the cases, n trades has been made in higher than 60 seconds. if this amount of trades has received
        # in less than 60 seconds, the asset is going to that 0.1 state.
        n = int(60 / np.percentile(diffs, quantile) * 10)
        n_trades[pair] = n
    return n_trades


def save_agg_trade_n_trades():
    client = Client()
    pairs = _get_usdt_pairs(client)
    n_trades = _get_n_trades_for_alts(pairs)
    with open('n_trades.pkl', 'wb') as f:
        pickle.dump(n_trades, f)
    return n_trades


def pars_args():
    parser = argparse.ArgumentParser(description='SAITA telegram bot.')

    parser.add_argument('--update_time_data',
                        help='Update the time-data at the start?',
                        action='store_true')
    parser.add_argument('--update_n_trades',
                        help='Update the time-data at the start?',
                        action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = pars_args()

    # Initiate a HistoricalDataHandler object
    data_handler = get_data_handler()

    # Update time-data
    if args.update_time_data:
        for base_currency, target_currency in itertools.product(BASE_CURRENCY_LIST, TARGET_CURRENCY_LIST):
            data_handler.update_time_data(base_currency,
                                          target_currency)

    if args.update_n_trades:
        N_TRADES = save_agg_trade_n_trades()

    # Create SAITA's processing unit, SAITA-core
    saita = get_saita()

    # Start telegram bot
    bot = start_telegram_bot(db_handler)

    # Start binance's websocket
    manager = start_binance_websocket_manager(True)

    # Start a loop to collect
    while True:
        now = datetime.datetime.now()
        next_call_dt = now + datetime.timedelta(hours=23 - now.hour,
                                                minutes=59 - now.minute,
                                                seconds=59 - now.second)
        seconds_to_next_midnight = float((next_call_dt - now).seconds)
        timer = threading.Timer(seconds_to_next_midnight,
                                collect_data,
                                args=(data_handler,))
        timer.start()
        timer.join()
