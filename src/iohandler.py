import os
import threading

import pandas as pd
from telegram import ParseMode

from src import COLLECTOR_MEMORY


mapper = {'kline_start_time': 't',
          'kline_close_time': 'T',
          'time_frame': 'i',
          'first_trade_id': 'f',
          'last_trade_id': 'L',
          'Open': 'o',
          'Close': 'c',
          'High': 'h',
          'Low': 'l',
          'Volume': 'v',  # base asset volume, use 'q' for target asset's volume
          'number_of_trades': 'n',
          'is_closed': 'x'}


def send_report(bot, users, report, plot_path):
    if plot_path:
        user = users[0]
        message = bot.send_message(chat_id=user.chat_id,
                                   text=report,
                                   parse_mode=ParseMode.MARKDOWN)
        with open(plot_path, 'rb') as photo:
            message = bot.send_photo(chat_id=user.chat_id,
                                     photo=photo,
                                     caption='Historical inference',
                                     reply_to_message_id=message.message_id)
        os.remove(plot_path)
        photo = message.photo[-1]
        for user in users[1:]:
            message = bot.send_message(chat_id=user.chat_id,
                                       text=report,
                                       parse_mode=ParseMode.MARKDOWN)
            bot.send_photo(chat_id=user.chat_id,
                           photo=photo,
                           caption='Historical inference',
                           reply_to_message_id=message.message_id)
    else:
        for user in users:
            bot.send_message(chat_id=user.chat_id,
                             text=report,
                             parse_mode=ParseMode.MARKDOWN)
            # logger


def create_candle(sub_candles):
    candle = {'Open': sub_candles[0]['Open'],
              'Close': sub_candles[-1]['Close'],
              'High': max([i['High'] for i in sub_candles]),
              'Low': min([i['Low'] for i in sub_candles]),
              'Volume': sum([i['Volume'] for i in sub_candles]),
              'DateTime': sub_candles[-1]['DateTime']}
    return candle


def sub_candle_collector(time_frame,
                         base_time_frame,
                         currency_pair,
                         saita,
                         bot,
                         db_handler,
                         children=None):
    print('I am {} collector for {}.'.format(time_frame.string, currency_pair))

    assert time_frame.minutes % base_time_frame.minutes == 0

    limit_length = int(time_frame.minutes / base_time_frame.minutes)

    memory = list()
    sub_candles = list()
    while True:
        sub_candle = (yield)
        sub_candles.append(sub_candle)
        if len(sub_candles) == limit_length:
            candle = create_candle(sub_candles)
            sub_candles = list()
            memory.append(candle)

            users = db_handler.get_matched_users(currency_pair, time_frame.string)
            if users:
                res = saita.generate_reports_time_based(currency_pair, time_frame, pd.DataFrame(memory))
                if res:
                    report, plot_path = res
                    print('sending report of {}/{} to {} users.'.format(currency_pair, time_frame.string, len(users)))
                    sender = threading.Thread(target=send_report, args=(bot, users, report, plot_path))
                    sender.start()
                    # print(reports)

            if len(memory) == COLLECTOR_MEMORY:
                memory.pop(0)

            if children:
                for child in children:
                    child.send(candle)


def read_binance_api():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    api_key = None
    api_secret = None
    with open(os.path.join(dir_path, 'binance.txt'), 'r') as file:
        for line in file.readlines():
            if line.startswith('API key'):
                api_key = line.split(' ')[-1]
            elif line.startswith('secret key'):
                api_secret = line.split(' ')[-1]
    return api_key, api_secret
