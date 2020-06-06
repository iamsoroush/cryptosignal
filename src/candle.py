import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
import talib

from src import PROFIT_INTERVALS, N_DERIVATIVES, N_NEAREST_NEIGHBORS


class CandleProcessor:
    """Candle Processing class.

    public methods:
        get_patterns_for_last_candle: Returns the candle-patterns found for the last time-step, as a dictionary.
        update_candles_df: Updates given dataframe by adding candle-pattern, highest_max and lowest_min for each
            profit_interval in PROFIT_INTERVALS, and derivatives of price and volume columns for each row.
        get_patterns_for_all_candles: Returns the candle-patterns found for each time-step(row) in provided dataframe.
        process_rank_last_candle: returns only one of 'Bearish', 'Bullish', 'Numb' for the last candle of the
            provided DataFrame.
        pattern_site_infernece: Returns the candle's type based on rankings provided at
            http://thepatternsite.com: Bullish, Bearish or Numb.
        majority_inference: Returns one of (Bullish, Bearish, Numb) based on counting the bullish and bearish patterns.
        historical_inference: Returns inference about the given patterns, based on provided historical data.
    """

    CANDLE_NAMES = ('Two Crows',
                    'Three Black Crows',
                    'Three Inside',
                    'Three-Line Strike',
                    'Three Outside',
                    'Three Stars In The South',
                    'Three Advancing White Soldiers',
                    'Abandoned Baby',
                    'Advance Block',
                    'Belt-hold',
                    'Breakaway',
                    'Closing Marubozu',
                    'Concealing Baby Swallow',
                    'Counterattack',
                    'Dark Cloud Cover',
                    'Doji',
                    'Doji Star',
                    'Dragonfly Doji',
                    'Engulfing Pattern',
                    'Evening Doji Star',
                    'Evening Star',
                    'Up/Down-gap side-by-side white lines',
                    'Gravestone Doji',
                    'Hammer',
                    'Hanging Man',
                    'Harami Pattern',
                    'Harami Cross Pattern',
                    'High-Wave Candle',
                    'Hikkake Pattern',
                    'Modified Hikkake Pattern',
                    'Homing Pigeon',
                    'Identical Three Crows',
                    'In-Neck Pattern',
                    'Inverted Hammer',
                    'Kicking',
                    'Kicking - bull/bear determined by the longer marubozu',
                    'Ladder Bottom',
                    'Long Legged Doji',
                    'Long Line Candle',
                    'Marubozu',
                    'Matching Low',
                    'Mat Hold',
                    'Morning Doji Star',
                    'Morning Star',
                    'On-Neck Pattern',
                    'Piercing Pattern',
                    'Rickshaw Man',
                    'Rising/Falling Three Methods',
                    'Separating Lines',
                    'Shooting Star',
                    'Short Line Candle',
                    'Spinning Top',
                    'Stalled Pattern',
                    'Stick Sandwich',
                    'Takuri (Dragonfly Doji with very long lower shadow)',
                    'Tasuki Gap',
                    'Thrusting Pattern',
                    'Tristar Pattern',
                    'Unique 3 River',
                    'Upside Gap Two Crows',
                    'Upside/Downside Gap Three Methods')

    FUNC_NAMES = ('CDL2CROWS',
                  'CDL3BLACKCROWS',
                  'CDL3INSIDE',
                  'CDL3LINESTRIKE',
                  'CDL3OUTSIDE',
                  'CDL3STARSINSOUTH',
                  'CDL3WHITESOLDIERS',
                  'CDLABANDONEDBABY',
                  'CDLADVANCEBLOCK',
                  'CDLBELTHOLD',
                  'CDLBREAKAWAY',
                  'CDLCLOSINGMARUBOZU',
                  'CDLCONCEALBABYSWALL',
                  'CDLCOUNTERATTACK',
                  'CDLDARKCLOUDCOVER',
                  'CDLDOJI',
                  'CDLDOJISTAR',
                  'CDLDRAGONFLYDOJI',
                  'CDLENGULFING',
                  'CDLEVENINGDOJISTAR',
                  'CDLEVENINGSTAR',
                  'CDLGAPSIDESIDEWHITE',
                  'CDLGRAVESTONEDOJI',
                  'CDLHAMMER',
                  'CDLHANGINGMAN',
                  'CDLHARAMI',
                  'CDLHARAMICROSS',
                  'CDLHIGHWAVE',
                  'CDLHIKKAKE',
                  'CDLHIKKAKEMOD',
                  'CDLHOMINGPIGEON',
                  'CDLIDENTICAL3CROWS',
                  'CDLINNECK',
                  'CDLINVERTEDHAMMER',
                  'CDLKICKING',
                  'CDLKICKINGBYLENGTH',
                  'CDLLADDERBOTTOM',
                  'CDLLONGLEGGEDDOJI',
                  'CDLLONGLINE',
                  'CDLMARUBOZU',
                  'CDLMATCHINGLOW',
                  'CDLMATHOLD',
                  'CDLMORNINGDOJISTAR',
                  'CDLMORNINGSTAR',
                  'CDLONNECK',
                  'CDLPIERCING',
                  'CDLRICKSHAWMAN',
                  'CDLRISEFALL3METHODS',
                  'CDLSEPARATINGLINES',
                  'CDLSHOOTINGSTAR',
                  'CDLSHORTLINE',
                  'CDLSPINNINGTOP',
                  'CDLSTALLEDPATTERN',
                  'CDLSTICKSANDWICH',
                  'CDLTAKURI',
                  'CDLTASUKIGAP',
                  'CDLTHRUSTING',
                  'CDLTRISTAR',
                  'CDLUNIQUE3RIVER',
                  'CDLUPSIDEGAP2CROWS',
                  'CDLXSIDEGAP3METHODS')

    CANDLE_RANKINGS = dict(CDL3LINESTRIKE_Bull=1, CDL3LINESTRIKE_Bear=2, CDL3BLACKCROWS_Bull=3, CDL3BLACKCROWS_Bear=3,
                           CDLEVENINGSTAR_Bull=4, CDLEVENINGSTAR_Bear=4, CDLTASUKIGAP_Bull=5, CDLTASUKIGAP_Bear=5,
                           CDLINVERTEDHAMMER_Bull=6, CDLINVERTEDHAMMER_Bear=6, CDLMATCHINGLOW_Bull=7,
                           CDLMATCHINGLOW_Bear=7, CDLABANDONEDBABY_Bull=8, CDLABANDONEDBABY_Bear=8,
                           CDLBREAKAWAY_Bull=10, CDLBREAKAWAY_Bear=10, CDLMORNINGSTAR_Bull=12, CDLMORNINGSTAR_Bear=12,
                           CDLPIERCING_Bull=13, CDLPIERCING_Bear=13, CDLSTICKSANDWICH_Bull=14, CDLSTICKSANDWICH_Bear=14,
                           CDLTHRUSTING_Bull=15, CDLTHRUSTING_Bear=15, CDLINNECK_Bull=17, CDLINNECK_Bear=17,
                           CDL3INSIDE_Bull=20, CDL3INSIDE_Bear=56, CDLHOMINGPIGEON_Bull=21, CDLHOMINGPIGEON_Bear=21,
                           CDLDARKCLOUDCOVER_Bull=22, CDLDARKCLOUDCOVER_Bear=22, CDLIDENTICAL3CROWS_Bull=24,
                           CDLIDENTICAL3CROWS_Bear=24, CDLMORNINGDOJISTAR_Bull=25, CDLMORNINGDOJISTAR_Bear=25,
                           CDLXSIDEGAP3METHODS_Bull=27, CDLXSIDEGAP3METHODS_Bear=26, CDLTRISTAR_Bull=28,
                           CDLTRISTAR_Bear=76, CDLGAPSIDESIDEWHITE_Bull=46, CDLGAPSIDESIDEWHITE_Bear=29,
                           CDLEVENINGDOJISTAR_Bull=30, CDLEVENINGDOJISTAR_Bear=30, CDL3WHITESOLDIERS_Bull=32,
                           CDL3WHITESOLDIERS_Bear=32, CDLONNECK_Bull=33, CDLONNECK_Bear=33, CDL3OUTSIDE_Bull=34,
                           CDL3OUTSIDE_Bear=39, CDLRICKSHAWMAN_Bull=35, CDLRICKSHAWMAN_Bear=35,
                           CDLSEPARATINGLINES_Bull=36, CDLSEPARATINGLINES_Bear=40, CDLLONGLEGGEDDOJI_Bull=37,
                           CDLLONGLEGGEDDOJI_Bear=37, CDLHARAMI_Bull=38, CDLHARAMI_Bear=72, CDLLADDERBOTTOM_Bull=41,
                           CDLLADDERBOTTOM_Bear=41, CDLCLOSINGMARUBOZU_Bull=70, CDLCLOSINGMARUBOZU_Bear=43,
                           CDLTAKURI_Bull=47, CDLTAKURI_Bear=47, CDLDOJISTAR_Bull=49, CDLDOJISTAR_Bear=51,
                           CDLHARAMICROSS_Bull=50, CDLHARAMICROSS_Bear=80, CDLADVANCEBLOCK_Bull=54,
                           CDLADVANCEBLOCK_Bear=54, CDLSHOOTINGSTAR_Bull=55, CDLSHOOTINGSTAR_Bear=55,
                           CDLMARUBOZU_Bull=71, CDLMARUBOZU_Bear=57, CDLUNIQUE3RIVER_Bull=60, CDLUNIQUE3RIVER_Bear=60,
                           CDL2CROWS_Bull=61, CDL2CROWS_Bear=61, CDLBELTHOLD_Bull=62, CDLBELTHOLD_Bear=63,
                           CDLHAMMER_Bull=65, CDLHAMMER_Bear=65, CDLHIGHWAVE_Bull=67, CDLHIGHWAVE_Bear=67,
                           CDLSPINNINGTOP_Bull=69, CDLSPINNINGTOP_Bear=73, CDLUPSIDEGAP2CROWS_Bull=74,
                           CDLUPSIDEGAP2CROWS_Bear=74, CDLGRAVESTONEDOJI_Bull=77, CDLGRAVESTONEDOJI_Bear=77,
                           CDLHIKKAKEMOD_Bull=82, CDLHIKKAKEMOD_Bear=81, CDLHIKKAKE_Bull=85, CDLHIKKAKE_Bear=83,
                           CDLENGULFING_Bull=84, CDLENGULFING_Bear=91, CDLMATHOLD_Bull=86, CDLMATHOLD_Bear=86,
                           CDLHANGINGMAN_Bull=87, CDLHANGINGMAN_Bear=87, CDLRISEFALL3METHODS_Bull=94,
                           CDLRISEFALL3METHODS_Bear=89, CDLKICKING_Bull=96, CDLKICKING_Bear=102,
                           CDLDRAGONFLYDOJI_Bull=98, CDLDRAGONFLYDOJI_Bear=98, CDLCONCEALBABYSWALL_Bull=101,
                           CDLCONCEALBABYSWALL_Bear=101, CDL3STARSINSOUTH_Bull=103, CDL3STARSINSOUTH_Bear=103,
                           CDLDOJI_Bull=104, CDLDOJI_Bear=104)

    def __init__(self):
        self.func_mapper = dict(zip(CandleProcessor.CANDLE_NAMES, CandleProcessor.FUNC_NAMES))
        self.name_mapper = dict(zip(CandleProcessor.FUNC_NAMES, CandleProcessor.CANDLE_NAMES))
        self.profit_intervals = PROFIT_INTERVALS
        self.n_volume_derivatives = N_DERIVATIVES

    def get_patterns_for_last_candle(self, df):

        """Returns the candle-patterns found for the last time-step.

        Note: returned patterns are in func_name format, i.e. CNDLDOJISTAR

        :arg df: Pandas DataFrame composed of 'Open', 'High', 'Low', 'Close'
        :returns patters: {'Bull': [pattern1_Bull, pattern2_Bull, ...], 'Bear': [pattern1_Bear, pattern2_Bear]}
        """

        patterns = dict(Bear=list(), Bull=list())
        for candle_name, func_name in self.func_mapper.items():
            c = getattr(talib, func_name)(df['Open'].values,
                                          df['High'].values,
                                          df['Low'].values,
                                          df['Close'].values)
            value = c[-1]
            name_to_append = func_name
            if func_name == 'CDLHIKKAKE' and np.abs(value) < 200:
                continue
            if np.abs(value) > 100:
                name_to_append += ' (Confirmed)'
            if value > 0:
                patterns['Bull'].append(name_to_append)
            elif value < 0:
                patterns['Bear'].append(name_to_append)
            else:
                continue
        return patterns

    def get_patterns_for_last_n_candles(self, df, n):

        """Returns the candle-patterns found for the last n time-steps.

        Note: returned patterns are in func_name format, i.e. CNDLDOJISTAR

        :arg df: Pandas DataFrame composed of 'Open', 'High', 'Low', 'Close'
        :returns patters: [{'Bull': [pattern1, pattern2, ...], 'Bear': [pattern1, pattern2]},
                           {'Bull': : [pattern1, pattern2, ...], 'Bear': [pattern1, pattern2]}, ...]
        """

        df_copy = df.copy()
        for candle_name, func_name in self.func_mapper.items():
            c = getattr(talib, func_name)(df['Open'].values,
                                          df['High'].values,
                                          df['Low'].values,
                                          df['Close'].values)
            df_copy[func_name] = c

        out = list()
        func_names = list(self.name_mapper.keys())
        for ind in range(-n, 0):
            patterns = dict(Bear=list(), Bull=list())
            series = df.iloc[ind][func_names]
            patterns['Bull'].extend(series[series > 0].index.tolist())
            patterns['Bear'].extend(series[series < 0].index.tolist())
            out.append(patterns)

        return out

    def update_candles_df(self, df):

        """Updates given dataframe by adding candle-pattern columns for each row, alongside the lowest/highest of n
            next steps which n is provided as self.profit_intervals.

        Note: make sure to do the same post-processing for new incoming candles when inferencing, i.e. add the same
            features (d(v) ...) to new candles at self._historical_inference and self._handle_small_historical_data.

        min-max columns ==> Highest-10-steps, Highest-n-steps, Lowest-10 steps

        :arg df: Pandas DataFrame composed of 'Open', 'High', 'Low', 'Close'
        """

        assert len(df) > 1, 'passed DataFrame must be of a length of bigger than 1'

        for candle_name, func_name in self.func_mapper.items():
            df[func_name] = getattr(talib, func_name)(df['Open'].values,
                                                      df['High'].values,
                                                      df['Low'].values,
                                                      df['Close'].values)
            if func_name == 'CDLHIKKAKE':
                df[np.abs(df[func_name]) < 200][func_name] = 0
        self._add_min_max(df, self.profit_intervals)
        self._add_price_derivative(df, n_d=3)
        self._add_volume_derivative(df, n_d=3)
        df.dropna(inplace=True)

    def get_patterns_for_all_candles(self, df):

        """Returns the candle-patterns found for each time-step.

        :arg df: Pandas DataFrame composed of 'Open', 'High', 'Low', 'Close', 'DateTime'
        :returns patters: {'DateTime1': {'Bull': [pattern1_Bull, pattern2_Bull, ...],
                                         'Bear': [pattern1_Bear, pattern2_Bear]},
                           'DateTime2': {'Bull': [],
                                         'Bear': []},
                            ...
                          }
        """

        assert len(df) > 1, 'passed DataFrame must be of a length of bigger than 1'

        patterns = {dt: dict(Bear=list(), Bull=list()) for dt in df['DateTime']}
        for candle_name, func_name in self.func_mapper.items():
            c = getattr(talib, func_name)(df['Open'].values,
                                          df['High'].values,
                                          df['Low'].values,
                                          df['Close'].values)
            for loc in np.where(c != 0)[0]:
                dt = df.iloc[loc]['DateTime']
                value = c[loc]
                if value > 0:
                    patterns[dt]['Bull'].append(candle_name)
                else:
                    patterns[dt]['Bear'].append(candle_name)
        return patterns

    def pattern_site_infernece(self, patterns):

        """Returns the candle's type based on rankings provided at http://thepatternsite.com: Bullish, Bearish or Numb

        :arg patterns: {'Bull': [pattern1_Bull, pattern2_Bull, ...], 'Bear': [pattern1_Bear, pattern2_Bear]}
        :returns bear_score and bull_score
        """

        ranks = list(CandleProcessor.CANDLE_RANKINGS.values())
        available_ranks = list(CandleProcessor.CANDLE_RANKINGS.keys())
        mean_rank = int(np.mean(ranks))
        bull_score = 0
        bear_score = 0
        for name in patterns['Bull']:
            # f_name = self.name_mapper[name]
            f_name = name + '_' + 'Bull'
            if f_name in available_ranks:
                bull_score += 1 / CandleProcessor.CANDLE_RANKINGS[f_name]
            else:
                bull_score += 1 / mean_rank
        for name in patterns['Bear']:
            # f_name = self.name_mapper[name]
            f_name = name + '_' + 'Bear'
            if f_name in available_ranks:
                bear_score += 1 / CandleProcessor.CANDLE_RANKINGS[f_name]
            else:
                bear_score += 1 / mean_rank
        if bull_score > bear_score:
            return 'Bullish'
        elif bear_score > bull_score:
            return 'Bearish'
        else:
            return 'Numb'

    def historical_inference(self, patterns, historical_df, target_candles, time_frame_minutes, plot=True):

        """Returns inference about the given patterns, based on provided historical data.

        :param target_candles: list of candles which the last one is the target candle.
        :arg patterns: output of self.get_patterns_for_last_candle()
        :arg historical_df: same as sample, historical candles.

        :returns res: pd.DataFrame of columns ['Highest_interval_step_change', 'Lowest_interval_step_change'] with
            intervals defined as self.profit_intervals, which its rows are statistical measures of matched samples
            found in historical_df.

            if res is None: <2 matched_samples has been found in provided historical data.
            else: matched_samples are fewer than 2 * N_NEAREST_NEIGHBORS, and the output is:
                ((matched_samples_high_max.describe(), matched_samples_low_min.describe()),
                 path_to_saved_violinplot)
        """

        if not patterns['Bull'] and not patterns['Bear']:
            return None
        else:
            matched_samples_indx = self._get_matched_sample_index(historical_df, patterns)
            matched_samples = self._handle_small_historical_data(historical_df,
                                                                 matched_samples_indx,
                                                                 target_candles,
                                                                 N_NEAREST_NEIGHBORS)
            if matched_samples is None:
                return None

            hcols = [i for i in matched_samples.columns if i.startswith('Highest')]
            lcols = [i for i in matched_samples.columns if i.startswith('Lowest')]

            highest_max_desc = matched_samples[hcols].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
            highest_max_desc.columns = [' '.join(i.split('-')[1:]) for i in highest_max_desc.columns]

            lowest_min_desc = matched_samples[lcols].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
            lowest_min_desc.columns = [' '.join(i.split('-')[1:]) for i in lowest_min_desc.columns]

            path_to_plot = None
            if plot and len(matched_samples) > 15:
                now = datetime.datetime.now()
                path_to_plot = '{}.png'.format(now.microsecond)
                fig = self._plot_dist(matched_samples, time_frame_minutes)
                fig.savefig(path_to_plot)
                plt.close()
            return (highest_max_desc, lowest_min_desc), path_to_plot

    def _handle_small_historical_data(self,
                                      historical_df,
                                      matched_samples_indx,
                                      target_candles,
                                      limit):

        """If returns None ==> matched samples are less than 2, else tha matched samples are less than
            MAX_HISTORICAL_DATA_SAMPLES"""

        n_matched_samples = len(matched_samples_indx)

        if n_matched_samples < 10:
            return None
        else:
            matched_samples = historical_df.loc[matched_samples_indx]
            matched_samples.dropna(inplace=True)
            if n_matched_samples < limit:
                return matched_samples
            else:
                cols = matched_samples.columns
                d_cols = [i for i in cols if i.startswith('d(')]

                self._add_price_derivative(target_candles, N_DERIVATIVES)
                self._add_volume_derivative(target_candles, N_DERIVATIVES)
                target_candles.dropna(inplace=True)
                if not np.any(target_candles):
                    return None

                means = matched_samples[d_cols].mean().values
                stds = matched_samples[d_cols].std().values

                x = (matched_samples[d_cols].values - means) / (stds + np.finfo(float).eps)
                y = np.expand_dims((target_candles[d_cols].values[-1] - means) / (stds + np.finfo(float).eps), axis=0)

                neighbor_indx = self._get_neighbor_indxs(x, y, limit)
                neighbors = matched_samples.iloc[neighbor_indx]
                return neighbors

    # @staticmethod
    # def generate_voilin_df(matched_samples):
    #     cols = matched_samples.columns
    #     high_cols = [item for item in cols if item.startswith('Highest')]
    #     low_cols = [item for item in cols if item.startswith('Lowest')]
    #
    #     data = list()
    #     labels = list()
    #     interval = list()
    #
    #     for p in PROFIT_INTERVALS:
    #         hc = [item for item in high_cols if int(item.split('-')[1]) == p][0]
    #         lc = [item for item in low_cols if int(item.split('-')[1]) == p][0]
    #
    #         h = matched_samples[hc].values
    #         l = matched_samples[lc].values
    #
    #         data.extend(np.concatenate([h, l]).tolist())
    #         labels.extend(['Highest maximum'] * len(h))
    #         labels.extend(['Lowest minimum'] * len(l))
    #         interval.extend(['{} next steps'.format(p)] * (len(h) + len(l)))
    #     return pd.DataFrame({'%Change': data,
    #                          'Label': labels,
    #                          '# Next steps': interval})

    @staticmethod
    def _plot_dist(matched_samples, time_frame_minutes):
        hcols = [i for i in matched_samples.columns if i.startswith('Highest')]
        lcols = [i for i in matched_samples.columns if i.startswith('Lowest')]

        highs = matched_samples[hcols]
        lows = matched_samples[lcols]

        fig, axes = plt.subplots(nrows=len(hcols),
                                 figsize=(16, 10),
                                 sharex=True,
                                 dpi=80)

        sns.set_style("darkgrid")

        for i in range(len(hcols)):
            ax = axes[i]
            ax.tick_params(axis='both', which='both', labelsize=12, labelbottom=True)
            h = highs[hcols[i]]
            l = lows[lcols[i]]
            sns.distplot(h, ax=ax, color='green', kde_kws={'bw': 0.2})
            sns.distplot(l, ax=ax, color='red', axlabel='%Change', kde_kws={'bw': 0.2})
            next_in_hours = int(hcols[i].split('-')[1]) * time_frame_minutes / 60
            ax.set_title('Next {:.2f} hours'.format(next_in_hours))
            ax.legend(['Highest Maximum', 'Lowest Minimum'])
        ax.set_xlabel('%Change')
        return fig

    @staticmethod
    def _get_neighbor_indxs(x, y, limit):

        """x is of shape (n_samples, n_features) and y is of shape (1, n_features)"""

        distances = np.squeeze(pairwise_distances(x, y))
        neighbor_indx = distances.argsort()[:limit]
        return neighbor_indx

    @staticmethod
    def _get_matched_sample_index(historical_df, patterns):
        if not patterns['Bull'] and not patterns['Bear']:
            return list()
        else:
            return historical_df.index[np.logical_and(np.all(historical_df[patterns['Bull']] > 0, axis=1),
                                                      np.all(historical_df[patterns['Bear']] < 0, axis=1))]

    @staticmethod
    def _get_matched_samples(data_df, patterns):
        res = data_df[np.all(data_df[patterns['Bull']] > 0, axis=1)]
        res = res[np.all(res[patterns['Bear']] < 0, axis=1)]
        return res

    @staticmethod
    def _add_min_max(df, profit_intervals):

        """Note: drop nan values using df.dropna(inplace=True)"""

        new_df = df[['Close', 'High', 'Low']]

        for i in range(1, max(profit_intervals) + 1):
            new_df['h_shifted_{}'.format(i)] = new_df['High'].shift(-i)
            new_df['l_shifted_{}'.format(i)] = new_df['Low'].shift(-i)

        # null_ind = new_df.index[new_df.isnull().any(axis=1)]

        cols = new_df.columns
        high_cols = [item for item in cols if item.startswith('h_shifted')]
        low_cols = [item for item in cols if item.startswith('l_shifted')]

        for p in profit_intervals:
            hc = [item for item in high_cols if int(item.split('_')[-1]) <= p]
            lc = [item for item in low_cols if int(item.split('_')[-1]) <= p]

            df['Highest-{}-steps'.format(p)] = 100 * (new_df[hc].max(axis=1) - new_df['Close']) / \
                                               (new_df['Close'] + np.finfo(float).eps)
            df['Lowest-{}-steps'.format(p)] = 100 * (new_df[lc].min(axis=1) - new_df['Close']) / \
                                              (new_df['Close'] + np.finfo(float).eps)

    @staticmethod
    def _add_price_derivative(df, n_d=3):
        p = df['Close'].values.astype(np.float)
        for d in range(n_d):
            p_padded = np.pad(p, (d + 1, 0), 'constant', constant_values=(np.nan, 0))[:len(p)]
            df['d(c){}'.format(d)] = (p - p_padded) / (p + np.finfo(float).eps)

    @staticmethod
    def _add_volume_derivative(df, n_d=3):
        v = df['Volume'].values.astype(np.float)
        for d in range(n_d):
            v_padded = np.pad(v, (d + 1, 0), 'constant', constant_values=(np.nan, 0))[:len(v)]
            df['d(v){}'.format(d)] = (v - v_padded) / (v + np.finfo(float).eps)

    @staticmethod
    def _add_close_based_profit(df, profit_interval):
        close = df['Close'].values.astype(np.float)
        c_padded = np.pad(close, (0, profit_interval), 'constant', constant_values=(0, np.nan))[:len(close)]
        df['Profit{}'.format(profit_interval)] = (c_padded[profit_interval:] - close) / close * 100
