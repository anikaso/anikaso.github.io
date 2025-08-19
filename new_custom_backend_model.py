import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz
import time
import requests
from textblob import TextBlob

# === CONFIG ===
FETCH_GOOGLE_TRENDS = False
API_KEY = "8GPqXzjJbrN4AC5exf4RLw2d8E1uH2GD"
YEARS_BACK = 5
timeframes = [('1h', 'hour', 1), ('4h', 'hour', 4), ('1d', 'day', 1), ('1w', 'week', 1)]
eastern = pytz.timezone('America/New_York')

# === HELPERS ===
def get_with_retry(url, params):
    resp = requests.get(url, params=params)
    if resp.status_code == 429:
        time.sleep(60)
        resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp

def fetch_corporate_actions(ticker, start_date):
    actions = pd.DataFrame()
    types = ['splits', 'dividends']
    for action_type in types:
        endpoint = f"https://api.polygon.io/v3/reference/{action_type}"
        params = {"ticker": ticker, "apiKey": API_KEY, "limit": 1000}
        resp = get_with_retry(endpoint, params)
        data = resp.json().get('results', [])
        if data:
            df = pd.DataFrame(data)
            if action_type == 'splits':
                df['event_date'] = pd.to_datetime(df['execution_date']).dt.date
                df = df[['event_date', 'split_from', 'split_to']]
                df['split_ratio'] = df['split_to'] / df['split_from']
            elif action_type == 'dividends':
                df['event_date'] = pd.to_datetime(df['ex_dividend_date']).dt.date
                df = df[['event_date', 'cash_amount']]
            actions = pd.merge(actions, df, how='outer', on='event_date') if not actions.empty else df
    return actions.rename(columns={'event_date': 'date'})

def fetch_fundamentals(ticker):
    endpoint = f"https://api.polygon.io/v3/reference/financials"
    params = {"ticker": ticker, "apiKey": API_KEY, "limit": 100}
    try:
        resp = get_with_retry(endpoint, params)
        data = resp.json().get('results', [])
        if data:
            df = pd.DataFrame(data)
            if 'as_of' in df.columns:
                df['report_date'] = pd.to_datetime(df['as_of']).dt.date
            elif 'reportDate' in df.columns:
                df['report_date'] = pd.to_datetime(df['reportDate']).dt.date
            else:
                df['report_date'] = pd.NaT
            return df
    except Exception as e:
        print(f"⚠️ Failed to fetch fundamentals: {e}")
    return pd.DataFrame()

def fetch_timeframe(ticker, label, span, multiplier, start_str, end_str=None,
                    news_agg_df=None, search_trends_df=None, corp_actions_df=None, fundamentals_df=None):
    if end_str:
        start_dt = eastern.localize(pd.to_datetime(start_str))
        start_ms = int(start_dt.astimezone(pytz.UTC).timestamp() * 1000)
        raw_end = pd.to_datetime(end_str)
        end_aware = eastern.localize(raw_end) if raw_end.tzinfo is None else raw_end
        end_ts_utc = int(end_aware.astimezone(pytz.UTC).timestamp() * 1000)
        end_dt_utc = end_aware.astimezone(pytz.UTC)
    else:
        raise ValueError("end_str is required in this pipeline")

    all_rows = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{span}/{start_ms}/{end_ts_utc}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}
    resp = get_with_retry(url, params)
    rows = resp.json().get('results', [])
    if not rows:
        return None
    all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert(eastern)
    suffix = f"_{label}"
    df = df.rename(columns={
        'o': 'open'+suffix, 'h': 'high'+suffix, 'l': 'low'+suffix, 'c': 'close'+suffix,
        'v': 'volume'+suffix, 'n': 'transactions'+suffix
    })
    df['close_lag1'+suffix] = df['close'+suffix].shift(1)
    df['date'] = df['timestamp'].dt.date

    if news_agg_df is not None and not news_agg_df.empty:
        df = df.merge(news_agg_df, on='date', how='left').fillna({'news_count':0, 'avg_sentiment':0})
    if search_trends_df is not None and not search_trends_df.empty:
        df = df.merge(search_trends_df, on='date', how='left').fillna({'search_volume':0})
    if corp_actions_df is not None and not corp_actions_df.empty:
        df = df.merge(corp_actions_df, on='date', how='left')
    if fundamentals_df is not None and not fundamentals_df.empty:
        df = df.merge(fundamentals_df, left_on='date', right_on='report_date', how='left').drop(columns=['report_date'], errors='ignore')

    df.drop(columns=['date'], inplace=True)
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first')
    return df

# === MAIN FUNCTION TO RUN EVERYTHING FOR A SYMBOL/TIMEFRAME ===
def run_full_pipeline(symbol, timeframe):
    asset = symbol.upper()
    company_name = asset

    now = datetime.now(eastern)
    earliest_start_date = (now - relativedelta(years=YEARS_BACK)).date()
    price_data_start_date = earliest_start_date
    trends_data_start_date = datetime.now().date() - timedelta(days=80)
    overall_end_date = datetime.now().date() - timedelta(days=1)

    # Fetch news
    resp = get_with_retry("https://api.polygon.io/v2/reference/news", {"ticker": asset, "limit": 500, "apiKey": API_KEY})
    news_df = pd.DataFrame(resp.json().get('results', []))
    agg_news_df = pd.DataFrame()
    if not news_df.empty:
        news_df['published_utc'] = pd.to_datetime(news_df['published_utc'], utc=True).dt.tz_convert(eastern)
        news_df['date'] = news_df['published_utc'].dt.date
        news_df['sentiment'] = news_df['title'].apply(lambda t: TextBlob(t).sentiment.polarity)
        agg_news_df = news_df.groupby('date').agg(news_count=('sentiment','count'),
                                                  avg_sentiment=('sentiment','mean')).reset_index()

    search_trends_df = pd.DataFrame()  # skipping Google Trends here
    corp_actions_df = fetch_corporate_actions(asset, price_data_start_date)
    fundamentals_df = fetch_fundamentals(asset)

    # Only fetch selected timeframe
    for label, span, mult in timeframes:
        if label == timeframe:
            end_datetime = datetime.now(eastern) - timedelta(minutes=15)
            end_str = end_datetime.isoformat()
            df = fetch_timeframe(asset, label, span, mult, price_data_start_date.isoformat(), end_str,
                                 news_agg_df=agg_news_df,
                                 search_trends_df=search_trends_df,
                                 corp_actions_df=corp_actions_df,
                                 fundamentals_df=fundamentals_df)
            if df is not None:
                df.to_csv(f"data_{asset}_{label}.csv", index=False)

# === LOAD + PREDICT FROM CSV ===
def load_and_predict_from_csv(symbol, timeframe):
    file_path = f"data_{symbol.upper()}_{timeframe}.csv"
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    close_col = f'close_{timeframe}'
    lag_col = f'close_lag1_{timeframe}'
    if lag_col not in df.columns or close_col not in df.columns:
        raise ValueError(f"Required columns not found in CSV: {lag_col}, {close_col}")

    df = df.dropna(subset=[lag_col, close_col])
    df['predicted_close'] = df[lag_col]
    df['actual_close'] = df[close_col]

    y_true = df['actual_close'].values
    y_pred = df['predicted_close'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = accuracy_score(np.sign(np.diff(y_true)), np.sign(np.diff(y_pred))) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['timestamp'], y_true, label='Actual Close', color='C0')
    ax.plot(df['timestamp'], y_pred, label='Predicted Close', color='C3')
    ax.set_title(f"{symbol.upper()} {timeframe.upper()} Price Prediction")
    ax.legend()
    ax.grid(True)

    predictions_df = df[['timestamp', 'actual_close', 'predicted_close']].copy()
    metrics = {'mae': mae, 'rmse': rmse, 'dir_acc': dir_acc}
    return fig, predictions_df, metrics
