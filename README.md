# CryptoSAITA

A Telegram bot which is an AI-based Trading Assistant (AITA).

## Install (ubuntu):
- `conda create -n crypto python=3.6.4`
- `conda activate crypto`
- `conda install pandas`
- `conda install matplotlib`
- `pip install mplfinance`
- `conda install seaborn`
- `conda install scikit-learn`
- `pip install peewee`
- `pip install ccxt`
- `pip install python-binance`
- `pip install python-telegram-bot --upgrade`
- `conda install -c quantopian ta-lib`
- `conda install requests`
- `conda install psutil`
- `conda install plotly`
- `conda install -c plotly plotly-orca`


## How to use?
After installing the required packages, place your *binance API Key and secret Key* on `binance.txt` file, your telegram bot token on `telegram.txt`, and your preferred timezone on `timezone.txt`. 

Now run the main file:
`python3 run main.py`

## Notes

- We assume that the Bot is running on a server with **UTC** timezone.
- Data-based reports and telegram-bot's language is **Persian**, you can change the reports and the bot's language on `saita_bot.py` and `saita.py` files. Sorry about that, I'm currently out of time to change the language to English.

## Sample Reports
*Pattern-based Report*


![alt text](patterns.jpeg "Patterns")


*Data-based Report (This is *Persian* :)) )*


![alt text](data_based.jpeg "Data-Based")