import datetime
import io

import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from src.candle import CandleProcessor
from src.data_handling import DataLoader
from src import N_DERIVATIVE_AGG_TRADE, N_DERIVATIVES, TIME_DATA_MEMORY_IN_DAYS
from src.utils import miliseconds_timestamp_to_str

report_time_based_consecutive = '''Report for *{}*@*{}*:

⬆️ *Bullish* patterns: {}
⬇️ *Bearish* patterns: {}

Open: *{:.2f}*
Close: *{:.2f}*
High: *{:.2f}*
Low: *{:.2f}*
Volume: *{:.2f}*
DateTime: _{}_

▫️ Inference based on [Patternsite](http://thepatternsite.com): *{}*

*Three consecutive {} patterns have been occured!*'''

report_time_based = '''Report for *{}*@*{}*:

⬆️ *Bullish* patterns: {}
⬇️ *Bearish* patterns: {}

    Open: *{:.2f}*
    Close: *{:.2f}*
    High: *{:.2f}*
    Low: *{:.2f}*
    Volume: *{:.2f}*
    DateTime: _{}_

▫️ Inference based on [Patternsite](http://thepatternsite.com): *{}*'''


class SAITA:

    def __init__(self):
        self.candle_processor = CandleProcessor()
        self.data_loader = DataLoader()
        self.plotter = Plotter()

    def generate_report_agg_trade(self, pair, n_trades, candles):

        """Generates report fro aggregated_trade candles.

        :param pair: ETHUSDT
        :param n_trades: number of trades each candle contains
        :param candles: list of candles, each one is a dictionary candle of ('Open': float, 'Close': float,
         'High': float, 'Low': float, 'Volume': float, 'Open Time': int(ms), 'Close Time': int(ms)]

        :returns report: it will be None if could'nt find any momentum3 change on price and volume, else a report to
            send to users.
        """

        last_candle = candles[-1]
        interval_sec = (last_candle['Close Time'] - last_candle['Open Time']) / 1000
        last_candle['Interval(s)'] = interval_sec

        if len(candles) < N_DERIVATIVE_AGG_TRADE + 1:
            return None

        else:
            if interval_sec < 60:
                second_last_candle = candles[-2]
                second_last_candle_interval = second_last_candle['Interval(s)']
                if interval_sec <= second_last_candle_interval:
                    p_change = (last_candle['Close'] - last_candle['Open']) / (
                            last_candle['Open'] + np.finfo(float).eps)
                    v_change = (last_candle['Volume'] - second_last_candle['Volume']) / (
                            second_last_candle['Volume'] + np.finfo(float).eps)
                    # inference = 'Bullish'
                    # if p_change < 0:
                    #     inference = 'Bearish'
                    # elif p_change == 0:
                    #     inference = None

                    caption = '''*Instant signal*: *{}* trades just have been made for *{}* in {} seconds.

Price change: {:.2f}%
Volume change: {:.2f}%'''.format(n_trades,
                                 pair,
                                 interval_sec,
                                 p_change * 100,
                                 v_change * 100)
                    path_to_plot = self.plotter.plot_candles_mplfinance_aggtrade(candles)
                    # buff = self.plotter.plot_candles_mplfinance(candles, inference, False)
                    return caption, path_to_plot

    def generate_reports_time_based(self, pair, time_frame, candles):

        """Generates reports.

        :param pair: ETHUSDT
        :param time_frame: obj of type src.TimeFrame
        :param candles: list of dictionaries with these keys (Open, Close, High, Low, Volume, DateTime(float))"""

        candles_df = pd.DataFrame(candles)
        res = self.get_reports_time_based(candles_df[['Open', 'High', 'Low', 'Close']])
        last_candle = candles[-1]
        last_candle['Inference'] = 'Numb'
        last_candle['Bullish Patterns'] = None
        last_candle['Bearish Patterns'] = None
        if res is None:
            return None

        patterns, candle_name_patterns, pattern_site_inference = res
        last_candle['Inference'] = pattern_site_inference

        if candle_name_patterns['Bull']:
            last_candle['Bullish Patterns'] = patterns['Bull']
            bulls = ' - '.join(['\n  🔘 _{}_'.format(item) for item in candle_name_patterns['Bull']])
        else:
            bulls = '_None_'

        if candle_name_patterns['Bear']:
            last_candle['Bearish Patterns'] = patterns['Bear']
            bears = ' - '.join(['\n  🔘 _{}_'.format(item) for item in candle_name_patterns['Bear']])
        else:
            bears = '_None_'
        formats = [pair,
                   time_frame.string,
                   bulls,
                   bears,
                   last_candle['Open'],
                   last_candle['Close'],
                   last_candle['High'],
                   last_candle['Low'],
                   last_candle['Volume'],
                   miliseconds_timestamp_to_str(last_candle['DateTime']).split('.')[0],
                   pattern_site_inference]

        historical_inference = self._get_historical_inference(candles_df[['Open', 'High', 'Low', 'Close', 'Volume']],
                                                              pair, time_frame, patterns)
        if historical_inference:
            (high_max_desc, low_min_desc), dist_plot_path = historical_inference
            col = high_max_desc.columns[0]
            next_hours = (time_frame.minutes * int(col.split(' ')[0])) / 60

            addon_report = '''
بر اساس مشاهدات شبیه به این الگو در {} روز گذشته:
    شانس اینکه قیمت در {:.2f} ساعت آینده بیش از {:.2f}% افت کند 10 درصد است.
    به احتمال 90 درصد قیمت در {:.2f} ساعت آینده حداقل {:.2f}% کاهش را تجربه میکند.
    شانس اینکه قیمت در {:.2f} ساعت آینده بیش از {:.2f}% افزایش یابد 10 درصد است.
    به احتمال 90 درصد قیمت در {:.2f} ساعت آینده حداقل {:.2f}% افزایش را تجربه میکند.
'''.format(TIME_DATA_MEMORY_IN_DAYS,
           next_hours,
           low_min_desc[col]['10%'],
           next_hours,
           low_min_desc[col]['90%'],
           next_hours,
           high_max_desc[col]['90%'],
           next_hours,
           high_max_desc[col]['10%'])
        else:
            addon_report = ''
            dist_plot_path = None

        if len(candles) > 2:
            if 'Inference' in candles[-2].keys() and 'Inference' in candles[-3].keys():
                if candles[-2]['Inference'] == pattern_site_inference and\
                        candles[-3]['Inference'] == pattern_site_inference:
                    path_to_plot = self.plotter.plot_candles_mplfinance(candles, pattern_site_inference, True)
                    formats.append(pattern_site_inference)
                    report = report_time_based_consecutive.format(*formats)
                    return report, path_to_plot, addon_report, dist_plot_path

        path_to_plot = self.plotter.plot_candles_mplfinance(candles, pattern_site_inference, False)
        report = report_time_based.format(*formats)
        return report, path_to_plot, addon_report, dist_plot_path

    @staticmethod
    def _prepare_historical_report(high_max, low_min, time_frame):
        hcols = high_max.columns
        formats = list()
        for col in hcols:
            step_minutes = int(col.split(' ')[0]) * time_frame.minutes

            l_10q = low_min[col]['10%']
            l_min = low_min[col]['min']

            h_10q = high_max[col]['90%']
            h_max = high_max[col]['max']

            formats.append("""
for the next *{:.2f}* hours:
    In 90% of cases the price dumps less than *{:.2f}%*, with a maximum dump of *{:.2f}%*
    In 90% of cases the price pumps less than *{:.2f}%*, with a maximum pump of *{:.2f}%*""".
                           format(step_minutes / 60,
                                  l_10q,
                                  l_min,
                                  h_10q,
                                  h_max))
        addon = '\n'.join(['{}' for _ in range(len(hcols))]).format(*formats)
        return addon

    def _get_historical_inference(self, candles_df, pair, time_frame, patterns):

        historical_df = self.data_loader.load_historical_time_data(pair,
                                                                   time_frame.string)
        if historical_df is not None and len(candles_df) > N_DERIVATIVES:
            historical_inference = self.candle_processor.historical_inference(patterns,
                                                                              historical_df,
                                                                              candles_df,
                                                                              time_frame.minutes)
        else:
            historical_inference = None
        return historical_inference

    def get_reports_time_based(self, candles_df):

        """Returns the processing results for given candle series.

        :param pair: if ETH/USDT ==> pair is ETHUSDT
        :param time_frame: time frame of the passed candles, type: TimeFrame
        :param candles_df: Pandas DataFrame composed of ['Open', 'High', 'Low', 'Close']

        :returns (candle_name_patterns, pattern_site_inference)
        Note: if return is None, then no patterns have been identified for the last candle of candles_df
         and you should consider returning 'Numb' as inference.
        """

        patterns = self.candle_processor.get_patterns_for_last_candle(candles_df)
        if not patterns['Bull'] and not patterns['Bear']:
            return None

        pattern_site_inference = self.candle_processor.pattern_site_infernece(patterns)

        # historical_df = self.data_loader.load_historical_time_data(pair,
        #                                                            time_frame.string)
        # historical_inference = None
        # if historical_df is not None and len(candles_df) > N_DERIVATIVES:
        #     historical_inference = self.candle_processor.historical_inference(patterns,
        #                                                                       historical_df,
        #                                                                       candles_df,
        #                                                                       time_frame.minutes,
        #                                                                       gen_violin_plots)

        candle_name_patterns = dict(Bull=list(), Bear=list())
        for fn in patterns['Bull']:
            candle_name_patterns['Bull'].append(self.candle_processor.name_mapper[fn])
        for fn in patterns['Bear']:
            candle_name_patterns['Bear'].append(self.candle_processor.name_mapper[fn])

        return patterns, candle_name_patterns, pattern_site_inference


class Plotter:

    def __init__(self):
        pass

    @staticmethod
    def plot_candles_mplfinance_aggtrade(candles):

        """Plots candles using mplfinance, saves the plot and returns the path to the .png file.

        :param candles: list of dictionaries with these keys: ['Open', 'High', 'Low', 'Close', 'Volume', 'DateTime'] and
         the 'DateTime' field must be a miliseconds timestamp.
        :param inference: the color of the last candle area will be determined based on this argument.
        :param three_same_patterns: whether to consider the three last candles for highlighting.

        :returns path_to_plot
        """

        sample_to_plot = pd.DataFrame(candles)
        sample_to_plot.index = pd.DatetimeIndex(sample_to_plot['DateTime'])
        now = datetime.datetime.now()
        path_to_plot = '{}.png'.format(now.microsecond)

        mpf.plot(sample_to_plot,
                 type='candle',
                 style='charles',
                 volume=True,
                 figscale=1.5,
                 savefig=path_to_plot,
                 show_nontrading=True)
        plt.close()
        return path_to_plot

    @staticmethod
    def plot_candles_mplfinance(candles, inference, three_same_patterns):

        """Plots candles using mplfinance, saves the plot and returns the path to the .png file.

        :param candles: list of dictionaries with these keys: ['Open', 'High', 'Low', 'Close', 'Volume', 'DateTime'] and
         the 'DateTime' field must be a miliseconds timestamp.
        :param inference: the color of the last candle area will be determined based on this argument.
        :param three_same_patterns: whether to consider the three last candles for highlighting.

        :returns path_to_plot
        """

        sample_to_plot = pd.DataFrame(candles)
        sample_to_plot.index = pd.DatetimeIndex(sample_to_plot['DateTime'])
        now = datetime.datetime.now()
        path_to_plot = '{}.png'.format(now.microsecond)
        if inference == 'Bullish':
            color = 'g'
        elif inference == 'Bearish':
            color = 'r'
        else:
            color = None

        if three_same_patterns:
            vlines_args = dict(vlines=sample_to_plot.index[-2], linewidths=160, alpha=0.3, colors=color)
        else:
            vlines_args = dict(vlines=sample_to_plot.index[-1], linewidths=40, alpha=0.3, colors=color)

        mpf.plot(sample_to_plot,
                 type='candle',
                 style='charles',
                 volume=True,
                 figscale=1.5,
                 savefig=path_to_plot,
                 show_nontrading=True,
                 vlines=vlines_args)
        plt.close()
        return path_to_plot

    @staticmethod
    def plot_candles_plotly(df):
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            row_heights=(0.8, 0.2),
                            vertical_spacing=0.02)

        fig.add_trace(go.Candlestick(x=df['DateTime'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     xaxis="x",
                                     yaxis="y",
                                     visible=True,
                                     showlegend=False),
                      row=1, col=1)
        colors = list()
        for row in df.iterrows():
            if row[1]['Close'] > row[1].Open:
                colors.append('green')
            else:
                colors.append('red')
        fig.add_trace(go.Bar(x=df['DateTime'],
                             y=df['Volume'],
                             showlegend=False,
                             marker_color=colors,
                             opacity=0.6),
                      row=2, col=1)

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)
        now = datetime.datetime.now()
        path_to_plot = '{}.png'.format(now.microsecond)
        fig.write_image(path_to_plot)
        return path_to_plot
