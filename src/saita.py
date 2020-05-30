from src.candle import CandleProcessor
from src.data_handling import DataLoader
from src.utils import miliseconds_timestamp_to_str
from src import TIME_DATA_MEMORY_IN_DAYS


class SAITA:

    def __init__(self):
        self.candle_processor = CandleProcessor()
        self.data_loader = DataLoader()

    def generate_reports_time_based(self, pair, time_frame, candles_df, gen_violin_plot=False):

        res = self.get_reports_time_based(pair, time_frame.string, candles_df, gen_violin_plot)
        if res is None:
            return None

        candle_name_patterns, pattern_site_inference, historical_inference = res

        if candle_name_patterns is None:
            return None

        if candle_name_patterns['Bull']:
            bulls = ' - '.join(['_{}_'.format(item) for item in candle_name_patterns['Bull']])
        else:
            bulls = '_None_'

        if candle_name_patterns['Bear']:
            bears = ' - '.join(['_{}_'.format(item) for item in candle_name_patterns['Bear']])
        else:
            bears = '_None_'
        last_candle = candles_df.iloc[-1]
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
        if historical_inference is None:
            addon = 'No matched pattern group found in the last {} days!'.format(TIME_DATA_MEMORY_IN_DAYS)
        else:
            (high_max, low_min), violin_plot_path = historical_inference
            addon = self._prepare_historical_report(high_max, low_min, time_frame)
        formats.append(addon)

        report = """Report for *{}*@*{}*:

🟢 *Bullish* patterns: {}
🔴 *Bearish* patterns: {}

    Open: *{}*
    Close: *{}*
    High: *{}*
    Low: *{}*
    Volume: *{}*
    DateTime: _{}_

▫️ Inference based on _Patternsite_: *{}*

▫️ Based on historical data: {}""".format(*formats)
        return report

    @staticmethod
    def _prepare_historical_report(high_max, low_min, time_frame):
        hcols = high_max.columns
        formats = list()
        for col in hcols:
            step_minutes = int(col.split(' ')[0]) * time_frame.minutes

            l_10q = low_min[col]['10%']
            l_min = low_min[col]['min']

            h_10q = high_max[col]['10%']
            h_max = high_max[col]['max']

            formats.append("""
for the next {} hours:
    In 90% of cases the price dumps less than *{}%*, with a maximum dump of *{}*
    In 90% of cases the price pumps less than *{}%*, with a maximum pump of *{}*""".
                           format(step_minutes / 60,
                                  l_10q,
                                  l_min,
                                  h_10q,
                                  h_max))
        addon = '\n'.join(['{}' for _ in range(len(hcols))]).format(*formats)
        return addon

    def get_reports_time_based(self, pair, time_frame, candles_df, gen_violin_plots):

        """Returns the processing results for given candle series.

        :param pair: if ETH/USDT ==> pair is ETHUSDT
        :param time_frame: time frame of the passed candles, type: TimeFrame
        :param candles_df: Pandas DataFrame composed of ['Open', 'High', 'Low', 'Close']

        :returns (patterns, pattern_site_inference, historical_inference)
        Note: if candle_name_patterns=None, then no patterns have inferenced for the last candle of candles_df and you
            should consider returning 'Numb' as inference.
        Note: If historical-time-data does not exists or no matched samples found, historical_inference=None
        """

        patterns = self.candle_processor.get_patterns_for_last_candle(candles_df)
        if not patterns['Bull'] and not patterns['Bear']:
            return None

        pattern_site_inference = self.candle_processor.pattern_site_infernece(patterns)

        historical_df = self.data_loader.load_historical_time_data(pair,
                                                                   time_frame)
        historical_inference = None
        if historical_df is not None:
            historical_inference = self.candle_processor.historical_inference(patterns,
                                                                              historical_df,
                                                                              candles_df,
                                                                              gen_violin_plots)

        candle_name_patterns = dict(Bull=list(), Bear=list())
        for fn in patterns['Bull']:
            candle_name_patterns['Bull'].append(self.candle_processor.name_mapper[fn])
        for fn in patterns['Bear']:
            candle_name_patterns['Bear'].append(self.candle_processor.name_mapper[fn])

        return candle_name_patterns, pattern_site_inference, historical_inference
