from time import sleep
from datetime import datetime

import numpy as np
import pandas as pd
from binance.client import Client
import ccxt

from src.utils import str_to_milisecond_timestamp, miliseconds_timestamp_to_str, get_logger


logger = get_logger(name=__name__, write_logs=True)


class CryptoTickFetcher:

    def __init__(self):
        self.binance = Client()
        self.aggregated_trades_columns = ['Aggregate tradeId', 'Price', 'Quantity', 'First tradeId',
                                          'Last tradeId', 'Timestamp', 'Was the buyer the maker?',
                                          'Was the trade the best price match?']
        self.aggregated_trades_max_limit = 1000

    def fetch_from_id_to_time(self,
                              symbol,
                              start_id,
                              end_time,
                              ticks=(100, 500, 1000, 5000, 10000)
                              ):

        """Fetch aggregated trade data using binance api's get_aggregate_trades starting from given trade_id.

        Note: use self.get_starting_id() to get the starting trade_id for your desired start_time.

        :arg symbol: ETHUSDT
        :arg start_id: e.g. 125415431
        :arg end_time: '2019-12-06 00:00:00.0'
        :arg ticks: tick counts to make candles from.

        :returns a list of pandas DataFrames, each one for a tick-per-candle, and contains each row as one candle with
            these columns: ['Open', 'High', 'Low', 'Close', 'Volume', 'DateTime', 'Open Time', 'Close Time'] and the last trade_id
        """

        now = datetime.now()
        end_time_in_ms = str_to_milisecond_timestamp(end_time)
        assert int(now.timestamp() * 1000) > end_time_in_ms, 'end_time is in the future! :)'

        # symbol = '{}{}'.format(base_currency.upper(), target_currency.upper())

        tick_bars = [list() for _ in ticks]
        all_trades = list()

        last_id = start_id
        done = False
        while True:
            trades = self._get_trades(symbol, last_id)

            time_stamps = np.array([float(tr['T']) for tr in trades])
            border = np.where(time_stamps >= end_time_in_ms)[0]
            if np.any(border):
                all_trades.extend(trades[:border[0]])
                done = True
            else:
                all_trades.extend(trades)
            last_id = all_trades[-1]['a'] + 1

            if len(all_trades) > ticks[-1]:
                to_process = all_trades[: ticks[-1]]
                all_trades = all_trades[ticks[-1]:]
                for trade_per_candle, tick_list in zip(ticks, tick_bars):
                    tick_list.extend(self._create_candles(to_process, trade_per_candle))
            elif done:
                for trade_per_candle, tick_list in zip(ticks[1:], tick_bars[1:]):
                    tick_list.extend(self._create_candles(all_trades, trade_per_candle))
            if done:
                break

        return [pd.DataFrame(item) for item in tick_bars], last_id

    def _fetch_trades(self, symbol, start_id, end_time_in_ms):
        last_id = start_id
        all_trades = list()
        logger.info('fetching agg_trades for {} ...'.format(symbol))
        while True:
            trades = self._get_trades(symbol, last_id)

            time_stamps = np.array([float(tr['T']) for tr in trades])
            border = np.where(time_stamps >= end_time_in_ms)[0]
            if np.any(border):
                all_trades.extend(trades[:border[0]])
                break
            else:
                all_trades.extend(trades)
                last_id = all_trades[-1]['a'] + 1
        logger.info('fetched {} trades for {}.'.format(len(all_trades), symbol))
        return self._generate_df_for_trades(all_trades)

    # def fetch(self,
    #           base_currency,
    #           target_currency,
    #           start_time,
    #           end_time,
    #           ticks=(100, 500, 1000, 5000, 10000)):
    #
    #     """Fetch trade data using binance api's get_aggregate_trades.
    #
    #     Each fetching process fetches 1000 aggregated trades.
    #     https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#market-data-endpoints
    #
    #
    #     :arg base_currency: e.g. ETH
    #     :arg target_currency: e.g. USDT
    #     :arg start_time: '2019-06-06 00:00:00.0'
    #     :arg end_time: '2019-12-06 00:00:00.0'
    #     :arg ticks: tick counts to make candles from.
    #
    #     :returns a list of pandas DataFrames, each one for a tick-per-candle, and contains each row as one candle with
    #         these columns: ['Open', 'High', 'Low', 'Close', 'Volume', 'DateTime', 'Open Time', 'Close Time']
    #
    #     """
    #
    #     assert start_time.endswith(' 00:00:00.0')
    #     assert end_time.endswith(' 00:00:00.0')
    #
    #     start_time_in_ms = str_to_milisecond_timestamp(start_time)
    #     end_time_in_ms = str_to_milisecond_timestamp(end_time)
    #
    #     start_id = self.get_starting_id(base_currency, target_currency, start_time_in_ms)
    #     last_id = start_id - 1
    #
    #     symbol = '{}{}'.format(base_currency.upper(), target_currency.upper())
    #
    #     tick_bars = [list() for _ in ticks]
    #     all_trades = list()
    #
    #     done = False
    #     while True:
    #         trades = self._get_trades(symbol, last_id + 1)
    #
    #         time_stamps = np.array([float(tr['T']) for tr in trades])
    #         border = np.where(time_stamps >= end_time_in_ms)[0]
    #         if np.any(border):
    #             all_trades.extend(trades[:border[0]])
    #             done = True
    #         else:
    #             all_trades.extend(trades)
    #
    #         if len(all_trades) > ticks[-1]:
    #             to_process = all_trades[: ticks[-1]]
    #             all_trades = all_trades[ticks[-1]:]
    #             for trade_per_candle, tick_list in zip(ticks, tick_bars):
    #                 tick_list.extend(self._create_candles(to_process, trade_per_candle))
    #         elif done:
    #             for trade_per_candle, tick_list in zip(ticks[1:], tick_bars[1:]):
    #                 tick_list.extend(self._create_candles(all_trades, trade_per_candle))
    #         if done:
    #             break
    #         else:
    #             last_id = all_trades[-1]['a']
    #
    #     return [pd.DataFrame(item) for item in tick_bars]

    def get_starting_id(self, symbol, time_in_ms):

        """Returns the starting trade_id given the starting time.

        :arg symbol: ETHUSDT
        :arg time_in_ms: start time-stamp in miliseconds (int).

        :returns aggregated_trade_id
        """

        # symbol = '{}{}'.format(base_currency.upper(), target_currency.upper())

        one_hour_in_ms = 60 * 60 * 1000
        trades = self.binance.get_aggregate_trades(symbol=symbol,
                                                   startTime=time_in_ms,
                                                   endTime=time_in_ms + one_hour_in_ms)
        if not trades:
            return None
        first_id = trades[0]['a']
        return first_id

    def _get_trades(self, symbol, last_id, max_retries=3, verbose=False):

        """Fetches self.aggregated_trades_max_limit trades starting from last_id.

        :arg symbol: ETHUSDT
        :arg last_id: the trade_id of starting trade.
        :arg max_retries: retires if any exception raises while fetching data.

        :returns trades: a list of dictionaries """

        num_retries = 0
        while True:
            try:
                trades = self.binance.get_aggregate_trades(symbol=symbol,
                                                           fromId=last_id,
                                                           limit=self.aggregated_trades_max_limit)
            except Exception as e:
                if num_retries < max_retries:
                    logger.exception('Exception occured while fetching: ', e.args[0])
                    num_retries += 1
                    sleep(5)
                    continue
                else:
                    raise e
            else:
                start_time = miliseconds_timestamp_to_str(trades[0]['T'])
                end_time = miliseconds_timestamp_to_str(trades[-1]['T'])
                if verbose:
                    logger.info('fetched {} trades from {} to {}'.format(len(trades),
                                                                         start_time,
                                                                         end_time))
                return trades

    def _generate_df_for_trades(self, trades):

        def _get_datetime(row):
            tstamp = row.Timestamp
            s = tstamp / 1000
            return datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.0')

        trades_df = pd.DataFrame(trades)
        trades_df.columns = self.aggregated_trades_columns
        trades_df = trades_df.astype({'Price': 'float',
                                      'Quantity': 'float'})
        trades_df['DateTime'] = trades_df.apply(_get_datetime, axis=1)
        return trades_df

    @staticmethod
    def _create_candles(trades, trade_per_candle):

        """Note: the returned Open Time and Close Time columns are of type int and as timestamps(ms)"""

        candles = list()

        if len(trades) >= trade_per_candle:
            for s in range(0, len(trades), trade_per_candle):
                e = s + trade_per_candle
                if e > len(trades):
                    break
                else:
                    slce = trades[s: e]
                    prices = np.array([item['p'] for item in slce], dtype=np.float)
                    open_price = prices[0]
                    close_price = prices[-1]
                    high_price = np.max(prices)
                    low_price = np.min(prices)
                    volume = np.array([item['q'] for item in slce], dtype=np.float).sum()
                    close_time = int(slce[-1]['T'])
                    open_time = int(slce[0]['T'])
                    candle = {'Open': open_price,
                              'High': high_price,
                              'Low': low_price,
                              'Close': close_price,
                              'Volume': volume,
                              'Open Time': open_time,
                              'Close Time': close_time}
                    candles.append(candle)
        return candles

    # @staticmethod
    # def get_candles(trades_df, trade_per_candle):
    #
    #     """Creates candle-sticks out of trades.
    #
    #     :arg trades_df: pd.DataFrame of each row an aggregated trade and the columns of :
    #         ['Aggregate tradeId', 'Price', 'Quantity', 'First tradeId',
    #          'Last tradeId', 'Timestamp', 'Was the buyer the maker?',
    #          'Was the trade the best price match?'].
    #          Export a raw list of trades to this format using get_df_for_trades
    #             method.
    #     :arg trade_per_candle: each candle will be composed of this count of trades.
    #
    #     :returns pd.DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume', 'DateTime']
    #     """
    #
    #     candles = list()
    #
    #     for s in range(0, len(trades_df), trade_per_candle):
    #         e = s + trade_per_candle
    #         if e > len(trades_df):
    #             break
    #         slce = trades_df.iloc[s: e]
    #         candle = {'Open': slce['Price'].iloc[0],
    #                   'High': slce['Price'].max(),
    #                   'Low': slce['Price'].min(),
    #                   'Close': slce['Price'].iloc[-1],
    #                   'Volume': slce['Quantity'].sum(),
    #                   'DateTime': datetime.fromtimestamp(slce.iloc[-1]['Timestamp'] / 1000).strftime(
    #                       '%Y-%m-%d %H:%M:%S.%f')}
    #         candles.append(candle)
    #     candles_df = pd.DataFrame(candles)
    #     return candles_df


class Binance:

    def __init__(self):
        self.exchange = ccxt.binance()
        self.markets = list(self.exchange.load_markets().keys())

    def get_data(self,
                 base_currency,
                 target_currency,
                 time_frame='5m',
                 from_time='2019-06-06 00:00:00.0',
                 to_time=None):
        data = self._download_data(base_currency, target_currency, time_frame, from_time, to_time)
        if data is not None:
            time_stamps = data[:, 0]
            diffs = time_stamps[1:] - time_stamps[:-1]
            if np.unique(diffs).size > 1:
                logger.warning('timing inconsistency has been found: sampling duration inconsistency.')
            df = self._get_df(data)
            return df
        else:
            return None

    def _download_data(self,
                       base_currency,
                       target_currency,
                       time_frame,
                       from_time,
                       to_time,
                       max_retries=3):

        pairs = [i for i in self.markets if i.startswith(base_currency.upper())]
        pair = [i for i in pairs if i.split('/')[-1] == target_currency.upper()]
        limit = 1000
        if pair:
            market = pair[0]

            from_time_in_ms = self.exchange.parse8601(from_time)
            if to_time is None:
                to_time_in_ms = datetime.now().timestamp() * 1000
            else:
                to_time_in_ms = self.exchange.parse8601(to_time)
            time_frame_duration_in_ms = self.exchange.parse_timeframe(time_frame) * 1000

            data = list()
            num_retries = 0
            while True:
                # logger.info('fetching data : ', miliseconds_timestamp_to_str(from_time_in_ms))
                try:
                    d = self.exchange.fetch_ohlcv(market, time_frame, from_time_in_ms, limit)
                except Exception as e:
                    if num_retries > max_retries:
                        raise e
                    else:
                        logger.exception(' Exception occured: ', e.args[0])
                        num_retries += 1
                        sleep(5)
                        continue
                else:
                    num_retries = 0
                if d:
                    time_stamps = (np.array(d)[:, 0]).astype(np.float)
                    border = np.where(time_stamps >= to_time_in_ms)[0]
                    if np.any(border):
                        data.extend(d[:border[0]])
                        break
                    else:
                        from_time_in_ms = int(d[-1][0] + time_frame_duration_in_ms)
                        data.extend(d)
                else:
                    break
            # print('data downloaded.')

            data = np.array(data)
            logger.info('{} ohlv data-points have been downloaded for {} market.'.format(data.shape[0], market))
        else:
            logger.warning('The {}/{} market does not exist.'.format(base_currency.upper(), target_currency.upper()))
            data = None
        return data

    def _get_df(self, data):
        header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame(data, columns=header).set_index('Timestamp')
        df['DateTime'] = df.apply(self._get_datetime, axis=1)
        return df

    @staticmethod
    def _get_datetime(row):
        tstamp = row.name
        s = tstamp / 1000.0
        return datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')