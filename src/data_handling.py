import os
from datetime import datetime
from datetime import timedelta as dt_timedelta

from src.fetching import Binance, CryptoTickFetcher
from src.utils import save_df, load_df, get_logger
from src.candle import CandleProcessor
from src import TIME_FRAMES, TICK_BARS, TIME_DATA_DIR, TICK_DATA_DIR, TIME_DATA_MEMORY_IN_DAYS


logger = get_logger(name=__name__, write_logs=True)


class BaseDataHandler:

    def __init__(self):
        self.time_frames = TIME_FRAMES
        self.tick_bars = TICK_BARS
        self.time_data_dir = TIME_DATA_DIR
        self.tick_data_dir = TICK_DATA_DIR

    @staticmethod
    def _dt_to_str(dt):
        """Datetime to str 2019-12-01 00:00:00.0"""

        return dt.strftime('%Y-%m-%d %H:%M:%S.0')

    @staticmethod
    def _ms_to_str(ms_timestamp):
        """Miliseconds timestamp to str 2019-12-01 00:00:00.0"""

        return datetime.fromtimestamp(ms_timestamp / 1000).isoformat()

    @staticmethod
    def _str_to_ms(str_dt):
        """String 2019-12-01 00:00:00.0 to miliseconds timestamp"""

        dt = datetime.strptime(str_dt, '%Y-%m-%d %H:%M:%S.0')
        dt_miliseconds = int(datetime.timestamp(dt)) * 1000
        return dt_miliseconds

    @staticmethod
    def _dt_to_ms(dt):
        """Datetime to miliseconds timestamp"""

        return int(dt.timestamp() * 1000)

    @staticmethod
    def _str_to_dt(str_time):
        """String to Datetime"""

        return datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S.0')


class DataLoader(BaseDataHandler):

    def __init__(self):
        super().__init__()

    def load_historical_time_data(self,
                                  pair,
                                  time_frame):
        sub_dir = os.path.join(self.time_data_dir, time_frame)
        df = None
        if os.path.isdir(sub_dir):
            existing_pairs = [i.split(' ')[0] for i in os.listdir(sub_dir)]
            if pair in existing_pairs:
                path = os.path.join(sub_dir, [i for i in os.listdir(sub_dir) if i.startswith(pair)][0])
                df = load_df(path)
            else:
                logger.warning('cant find any historical time-data for {}/{}'.format(pair, time_frame))
        else:
            logger.warning('data_dir does not exist.')
        return df


class TimeDataHandler(BaseDataHandler):

    def __init__(self):
        super().__init__()
        self.binance = Binance()
        self.candle_processor = CandleProcessor()

    def update_time_data(self, base_currency, target_currency):

        """Downloads and saves the data from last_n_days ago until the last monday.

        :param base_currency: base currency, ETH
        :param target_currency: target currency, USDT
        :param last_n_days: for last 6 months, pass 180
        """

        if base_currency == target_currency:
            logger.warning('skipping {}{}'.format(base_currency, target_currency))
            return

        base_currency = base_currency.upper()
        target_currency = target_currency.upper()
        pair = '{}{}'.format(base_currency, target_currency)

        now = datetime.now()
        end_dt = now - dt_timedelta(days=now.weekday(),
                                    hours=now.hour,
                                    minutes=now.minute,
                                    seconds=now.second)
        start_dt = end_dt - dt_timedelta(days=TIME_DATA_MEMORY_IN_DAYS)

        start_time = self._dt_to_str(start_dt)
        end_time = self._dt_to_str(end_dt)

        if not os.path.exists(self.time_data_dir):
            os.mkdir(self.time_data_dir)

        for time_frame in self.time_frames:
            # logger.info('==================================================================================')
            tf_str = time_frame.string
            logger.info('downloading historical time data for {}/{} from {} to {}'.format(pair,
                                                                                          tf_str,
                                                                                          start_time,
                                                                                          end_time))
            df = self.binance.get_data(base_currency, target_currency, tf_str, start_time, end_time)
            self.candle_processor.update_candles_df(df)

            tf_dir = os.path.join(self.time_data_dir, tf_str)
            if not os.path.exists(tf_dir):
                os.mkdir(tf_dir)

            # path = os.path.join(tf_dir, '{} {} {} {}.csv'.format(pair,
            #                                                      time_frame,
            #                                                      start_time,
            #                                                      end_time))
            path = os.path.join(tf_dir, '{} {}.csv'.format(pair, tf_str))
            save_df(df, path)
            logger.info('data for {}/{} saved to {}'.format(pair, tf_str, path))

    def save_historical_time_data(self,
                                  base_currency,
                                  target_currency,
                                  start_time,
                                  end_time):

        """Fetches time-based data from Binance using ccxt, generaets ochlcv-candle dataframe and saves that to data_dir.

        :arg base_currency: ETH
        :arg target_currency: USDT
        :arg start_time: '2019-12-01 00:00:00.0'
        :arg end_time: '2020-4-01 00:00:00.0'
        """

        if not os.path.exists(self.time_data_dir):
            os.mkdir(self.time_data_dir)

        for time_frame in [i.string for i in self.time_frames]:
            logger.info('==================================================================================')
            print('Pair: {}{}   timeframe: {}'.format(base_currency.upper(), target_currency.upper(), time_frame))
            df = self.binance.get_data(base_currency.upper(), target_currency.upper(), time_frame, start_time, end_time)
            self.candle_processor.update_candles_df(df)

            tf_dir = os.path.join(self.time_data_dir, time_frame)
            if not os.path.exists(tf_dir):
                os.mkdir(tf_dir)

            path = os.path.join(tf_dir,
                                '{}{} {} {} {}.csv'.format(base_currency.upper(),
                                                           target_currency.upper(),
                                                           time_frame,
                                                           start_time.split('.')[0].replace(':', '-').replace(' ', '_'),
                                                           end_time.split('.')[0].replace(':', '-').replace(' ', '_')))
            save_df(df, path)
            print('saved to ', path)


class TickDataHandler(BaseDataHandler):

    def __init__(self):
        super().__init__()
        self.tick_fetcher = CryptoTickFetcher()
        self.candle_processor = CandleProcessor()

    def save_historical_data(self,
                             base_currency,
                             target_currency,
                             start_time,
                             end_time):

        """Fetches historical aggregated trade data from binance, and saves them each 7 days.

        Note: both start_time and end_time must have time values of 0.

        :arg base_currency: ETH
        :arg target_currency: USDT
        :arg start_time: '2019-12-01 00:00:00.0'
        :arg end_time: '2020-4-01 00:00:00.0'
        """

        start_dt = self._str_to_dt(start_time)

        if start_dt.weekday() != 0:
            first_start_dt = start_dt + dt_timedelta(days=7 - start_dt.weekday())
        else:
            first_start_dt = start_dt
        end_dt = self._str_to_dt(end_time)

        self._check_datetime_inputs(first_start_dt, end_dt)

        symbol = '{}{}'.format(base_currency.upper(), target_currency.upper())

        start_time_in_ms = self._dt_to_ms(first_start_dt)
        first_trade_id = self.tick_fetcher.get_starting_id(symbol, start_time_in_ms)
        start_id = first_trade_id

        delta = end_dt - first_start_dt

        for add_days in range(7, delta.days, 7):
            new_end_dt = first_start_dt + dt_timedelta(days=add_days)
            new_end_str = self._dt_to_str(new_end_dt)
            new_start_dt = first_start_dt + dt_timedelta(days=add_days - 7)
            new_start_str = self._dt_to_str(new_start_dt)

            print('\n===========================================================================')
            print('pair: {}{}'.format(base_currency.upper(), target_currency.upper()))
            print('from {} to {}\n'.format(new_start_str, new_end_str))

            if self._exists(self.tick_data_dir, new_start_str):
                print('Skipping {} to {}'.format(new_start_str, new_end_str))
                continue

            tick_data, last_id = self.tick_fetcher.fetch_from_id_to_time(base_currency,
                                                                         target_currency,
                                                                         start_id,
                                                                         new_end_str,
                                                                         self.tick_bars)

            self._write_data(base_currency, target_currency, new_start_str, new_end_str, tick_data)
            start_id = last_id + 1

    def save_last_week_data(self, base_currency, target_currency, end_time):

        """Saves the last week's data, from weekday0 to last weekday0"""

        end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f')

        assert not (end_dt.hour or end_dt.minute or end_dt.second), 'change the end_time'
        assert end_dt.weekday() == 0, 'end date must be the weeks first day'

        now = datetime.now()
        end_time_in_ms = self._str_to_ms(end_time)
        assert int(now.timestamp() * 1000) > end_time_in_ms, 'end_time is in the future! :)'

        start_dt = end_dt - dt_timedelta(days=7)
        start_time = self._dt_to_str(start_dt)

        start_time_in_ms = int(start_dt.timestamp() * 1000)

        symbol = '{}{}'.format(base_currency.upper(), target_currency.upper())
        first_trade_id = self.tick_fetcher.get_starting_id(symbol, start_time_in_ms)

        tick_data, last_id = self.tick_fetcher.fetch_from_id_to_time(base_currency,
                                                                     target_currency,
                                                                     first_trade_id,
                                                                     end_time,
                                                                     self.tick_bars)

        self._write_data(base_currency, target_currency, start_time, end_time, tick_data)

    def _fetch_last_2weeks_data(self, pair):
        now = datetime.now() - dt_timedelta(hours=6)
        start_dt = now - dt_timedelta(days=14)

        start_time_in_ms = int(start_dt.timestamp() * 1000)

        first_trade_id = self.tick_fetcher.get_starting_id(pair, start_time_in_ms)
        if first_trade_id is None:
            return list()
        tick_data = self.tick_fetcher._fetch_trades(pair,
                                                    first_trade_id,
                                                    self._dt_to_ms(now))
        return tick_data

    # def save_historical_trade_data(self, base_currency, target_currency, from_time, to_time):
    #
    #     """Fetches historical aggregated trade data from binance, makes candles out of them and saves candles plus
    #         patterns generated for them."""
    #
    #     tick_data, last_id = self.tick_fetcher.fetch(base_currency, target_currency, from_time, to_time, self.tick_bars)
    #     self._save_tick_data(base_currency, target_currency, from_time, to_time, tick_data)

    @staticmethod
    def _check_datetime_inputs(start_dt,
                               end_dt):

        """Checks inputs for self.save_7day_historical_trade_data method."""

        assert not (start_dt.hour or start_dt.minute or start_dt.second), 'change the start_time'
        assert not (end_dt.hour or end_dt.minute or end_dt.second), 'change the end_time'

        time_delta = end_dt - start_dt
        assert time_delta.days >= 7, 'start_time to end_time must be greater thn 7 days!'

    @staticmethod
    def _exists(data_dir, start_time):

        if os.path.exists(data_dir):
            sub_dirs = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
            all_dirs = [False for _ in range(len(sub_dirs))]
            for i, d in enumerate(sub_dirs):
                file_names = os.listdir(d)
                time_formatted = start_time.split('.')[0].replace(':', '-').replace(' ', '_')
                for f in file_names:
                    if time_formatted in f.split(' '):
                        all_dirs[i] = True
                        break

            if all(all_dirs):
                return True
        return False

    def _write_data(self, base_currency, target_currency, start_time, end_time, tick_data):

        """Saves list of dataframe data in subdirectories of self.tick_bars at data_dir."""

        if not os.path.exists(self.tick_data_dir):
            os.mkdir(self.tick_data_dir)

        for tick_per_candle, tick_df in zip(self.tick_bars, tick_data):

            sub_dir = os.path.join(self.tick_data_dir, '{}_tick_per_candle'.format(tick_per_candle))
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            path = os.path.join(sub_dir,
                                '{}{} {} {} {}.csv'.format(base_currency.upper(),
                                                           target_currency.upper(),
                                                           tick_per_candle,
                                                           start_time.split('.')[0].replace(':', '-').replace(' ', '_'),
                                                           end_time.split('.')[0].replace(':', '-').replace(' ', '_')))
            save_df(tick_df, path)
            print('saved to ', path)
