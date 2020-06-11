import os
import threading

import numpy as np
from telegram import ParseMode

from src import COLLECTOR_MEMORY, AGG_TRADE_COLLECTOR_MEMORY, GROUP_CHAT_ID


def send_report(bot, users, report, plot_path, hist_inference_report, dist_plot_path):
    user = users[0]
    with open(plot_path, 'rb') as photo:
        message = bot.send_photo(chat_id=user.chat_id,
                                 photo=photo,
                                 caption=report,
                                 parse_mode=ParseMode.MARKDOWN)
    os.remove(plot_path)
    photo_candles = message.photo[-1]
    if dist_plot_path:
        with open(dist_plot_path, 'rb') as photo:
            message = bot.send_photo(chat_id=user.chat_id,
                                     photo=photo,
                                     caption=hist_inference_report,
                                     reply_to_message_id=message.message_id,
                                     parse_mode=ParseMode.MARKDOWN)
        os.remove(dist_plot_path)
        photo_dist = message.photo[-1]

    for user in users[1:]:
        message = bot.send_photo(chat_id=user.chat_id,
                                 photo=photo_candles,
                                 caption=report,
                                 parse_mode=ParseMode.MARKDOWN)
        if dist_plot_path:
            bot.send_photo(chat_id=user.chat_id,
                           photo=photo_dist,
                           caption=hist_inference_report,
                           reply_to_message_id=message.message_id,
                           parse_mode=ParseMode.MARKDOWN)


    # if plot_path:
    #     user = users[0]
    #     message = bot.send_message(chat_id=user.chat_id,
    #                                text=report,
    #                                parse_mode=ParseMode.MARKDOWN,
    #                                disable_web_page_preview=True)
    #     with open(plot_path, 'rb') as photo:
    #         message = bot.send_photo(chat_id=user.chat_id,
    #                                  photo=photo,
    #                                  caption='Historical inference',
    #                                  reply_to_message_id=message.message_id)
    #     os.remove(plot_path)
    #     photo = message.photo[-1]
    #     for user in users[1:]:
    #         message = bot.send_message(chat_id=user.chat_id,
    #                                    text=report,
    #                                    parse_mode=ParseMode.MARKDOWN)
    #         bot.send_photo(chat_id=user.chat_id,
    #                        photo=photo,
    #                        caption='Historical inference',
    #                        reply_to_message_id=message.message_id)
    # else:
    #     for user in users:
    #         bot.send_message(chat_id=user.chat_id,
    #                          text=report,
    #                          parse_mode=ParseMode.MARKDOWN,
    #                          disable_web_page_preview=True)


def send_agg_trade_report(bot, caption, plot_path):
    with open(plot_path, 'rb') as photo:
        bot.send_photo(chat_id=GROUP_CHAT_ID,
                       photo=photo,
                       caption=caption,
                       parse_mode=ParseMode.MARKDOWN)
    os.remove(plot_path)

    # bot.send_message(chat_id=GROUP_CHAT_ID,
    #                  text=report,
    #                  parse_mode=ParseMode.MARKDOWN)


def _create_candle(sub_candles):
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

    """Collects sub-candles and send messages if a candle has been created.

    Note: each yield is a dictionary of:
        {'Open': float,
         'Close': float,
         'High': float,
         'Low': float,
         'Volume': float,
         'DateTime': float(ms).

    :param time_frame: object of TimeFrame
    :param base_time_frame: object of TimeFrame
    :param currency_pair: 'ETHUSDT'
    :param saita: an object of saita.SAITA for processing the candle and generating report
    :param bot: Telegram.Bot object for sending messages
    :param db_handler: object of DBHandler for fetching data from database.
    :param children: a list of collectors to send the candles from this collector to them."""

    print('I am {} collector for {}.'.format(time_frame.string, currency_pair))

    assert time_frame.minutes % base_time_frame.minutes == 0

    limit_length = int(time_frame.minutes / base_time_frame.minutes)

    memory = list()
    sub_candles = list()
    while True:
        sub_candle = (yield)
        sub_candles.append(sub_candle)
        if len(sub_candles) == limit_length:
            candle = _create_candle(sub_candles)
            sub_candles = list()
            memory.append(candle)

            users = db_handler.get_matched_users(currency_pair, time_frame.string)
            if users:
                res = saita.generate_reports_time_based(currency_pair, time_frame, memory)
                if res:
                    report, plot_path, hist_inference_report, dist_plot_path = res
                    print('sending report of {}/{} to {} users.'.format(currency_pair, time_frame.string, len(users)))
                    sender = threading.Thread(target=send_report,
                                              args=(bot,
                                                    users,
                                                    report,
                                                    plot_path,
                                                    hist_inference_report,
                                                    dist_plot_path),
                                              daemon=True)
                    sender.start()
                    # print(reports)

            if len(memory) == COLLECTOR_MEMORY:
                memory.pop(0)

            if children:
                for child in children:
                    child.send(candle)


def _create_agg_candle(trades):
    selling_ratio = np.mean(trades[:, 3])  # 1 ==> Bearish, 0 ==> Bullish
    candle = {'Open': trades[0][1],
              'Close': trades[-1][1],
              'High': trades[:, 1].max(),
              'Low': trades[:, 1].min(),
              'Volume': trades[:, 2].sum(),
              'Open Time': float(trades[0][0]),
              'Close Time': float(trades[-1][0]),
              'DateTime': float(trades[-1][0]),
              'SellRatio': selling_ratio}
    return candle


def agg_trade_coroutine(currency_pair, base_n_trades, n_trades, saita, bot, db_handler, children=None):

    """In each yield, gents a dictionary of :
        {'trade_time': float(ms), 'price': float, 'volume': float, 'maker_is_buyer': float}

    Note: for base coroutine, give base_n_trades=1
    """

    assert n_trades % base_n_trades == 0

    memory = list()
    trades = np.zeros((n_trades, 3), dtype=np.float)
    pointer = 0
    while True:
        trade = (yield)
        trades[pointer] = trade
        pointer += 1
        if pointer == n_trades // base_n_trades:

            # Make candle
            candle = _create_agg_candle(trades)

            # Initialize
            trades = np.zeros((n_trades, 3))
            pointer = 0

            # Process
            memory.append(candle)
            report = saita.generate_report_agg_trade(currency_pair, n_trades, memory)
            if report:
                caption, plot_path = report
                print('sending agg_trade report of {}/{} to the group.'.format(currency_pair, n_trades))
                sender = threading.Thread(target=send_agg_trade_report, args=(bot, caption, plot_path), daemon=True)
                sender.start()
            if len(memory) > AGG_TRADE_COLLECTOR_MEMORY:
                memory.pop(0)

            # Send to children
            if children:
                for child in children:
                    child.send(candle)
