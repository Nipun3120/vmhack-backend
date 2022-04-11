import requests
import os
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import joblib
import twint

from scipy.stats import pearsonr
from flair.models import TextClassifier
from flair.data import Sentence
import tensorflow as tf
from nsepy import get_history
from datetime import datetime, timedelta
# import datetime
import yfinance as yf
from dotenv import load_dotenv


# MODELS
MODEL_APPLE = tf.keras.models.load_model('models/apple_savedmodel/')
MODEL_MICROSOFT = tf.keras.models.load_model('models/microsoft_savedmodel/')
MODEL_NIFTY = tf.keras.models.load_model('models/nifty_savedmodel/')
MODEL_DOW = tf.keras.models.load_model('models/dow_savedmodel/')


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def clean(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'pic.twitter\S+', ' ', text)
    # text = decontracted(text)
    text = re.sub(r'\(([^)]+)\)', " ", text)
    text = text.replace('etmarkets', ' ').replace('marketupdates', ' ').replace('newsalert', ' ').replace('ndtv', ' ').replace('moneycontrol', ' ').replace('here is why', ' ')
    text = text.replace('marketsupdate', ' ').replace('biznews', ' ').replace('click here', ' ').replace('live updates', ' ').replace('et now', ' ')
    text = re.sub(r'[^a-zA-Z0-9 ]+', ' ', text)
    text = re.sub(r' \w{1,2}_', ' ', text)
    text = re.sub('\s+',' ', text)
    return text


analyzer = SentimentIntensityAnalyzer()
def sentiments(text):
    return analyzer.polarity_scores(text)['compound']

classifier = TextClassifier.load('en-sentiment')
def flair_sentiments(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    return sentence.labels[0].score if sentence.labels[0].value == 'POSITIVE' else -sentence.labels[0].score

def getSentimentsOfTweets(tweets):
    return [flair_sentiments(x) for x in tweets]

def weekly_correlation(df):
    a = np.array(df['close_price'])
    b = np.array(df['score'])
    b_adjusted = np.append(b[1:],1)
    # b_adjusted, b

    return pearsonr(a,b_adjusted)

def stripdate(date):
    return date.split(' ')[0]
# abbe krne de do min acha oki
# change all occurences kara hai
#thode changes karne padenge, isme sab kaam ho rha ab, short mein. tumhara front end pe bhi yahi se bhej dena
tweets_df = []
config = []
async def getTweetsWithSentiments(city, keyword, sinceDate, untilDate):

    global config

    config.append(twint.Config())
    config[-1].Near = city
    config[-1].Search = keyword
    config[-1].Lang = "en"
    config[-1].Since = str(sinceDate)
    config[-1].Until = str(untilDate)
    # config.Until = str(untilDate.date())
    config[-1].Popular_tweets = True
    config[-1].Pandas = True
    config[-1].limit = 100
    config[-1].Hide_output = True

    twint.run.Search(config[-1])

    global tweets_df

    tweets_df.append(twint.storage.panda.Tweets_df)

    print('keyword------------->', keyword)
    print(tweets_df[-1][['tweet', 'date']])

    tweets_df[-1]['date'] = tweets_df[-1]['date'].apply(stripdate)
    tweets_df[-1]['tweet_clean'] = tweets_df[-1]['tweet'].apply(clean)
    tweets_df[-1] = tweets_df[-1].sort_values(['date','nreplies','nlikes','nretweets'], ascending=[False, False, False, False])[['date', 'tweet', 'tweet_clean', 'nlikes','nretweets', 'nreplies']].reset_index()

    a = list(tweets_df[-1].columns)
    a.append('sentiment')
    df_final = pd.DataFrame(columns = a)
    for date in list(tweets_df[-1]['date'].unique()):
        df_temp = pd.DataFrame(columns = tweets_df[-1].columns)
        df_temp = pd.concat([df_temp,tweets_df[-1][tweets_df[-1]['date']==date][:20]])

        df_temp['sentiment'] = tweets_df[-1][tweets_df[-1]['date']==date][:20]['tweet'].apply(flair_sentiments)
        df_final = pd.concat([df_final,df_temp])

    df_final = df_final.sort_values(['date', 'sentiment'], ascending= [False, False])

    sentiments = []
    prices = []

    stock_prices = await fetchStockPrices(untilDate, untilDate - timedelta(days=10), keyword)
    stock_prices['open-close'] = stock_prices ['Open'] - stock_prices['Close']

    s = set(list(df_final['date']))
    n = set([x for x in list(stock_prices['Date'])])


    common_dates_1 = [x for x in s if x in n]


    for i in common_dates_1:
        sentiments.append(df_final[df_final['date'] == i]['sentiment'].mean())
        prices.append(stock_prices[stock_prices['Date'] == i]['open-close'].values[0])
    
    # prices = np.append(np.nan, np.array(prices)[:-1])
    # mask = np.isnan(prices)
    # idx = np.where(~mask,np.arange(mask.shape[1]),0)
    # np.maximum.accumulate(idx,axis=1, out=idx)
    # prices = prices[idx]

    return df_final, pearsonr(np.append(np.mean(prices), np.array(prices)[:-1]), np.array(sentiments))
    return df_final, pearsonr(prices, np.array(sentiments))


def getTopTweets(tweets, sentiments):
    df = pd.DataFrame(list(zip(tweets, sentiments)), columns=['tweets', 'sentiments'])
    df = df.sort_values('sentiments')
    return df

def correct_date_fmt(date):
    return str(date.strftime('%Y-%m-%d'))
# returns a list of prices (okay?)
async def fetchStockPrices(currDate, prevDate, keyword):
    
    if keyword == "MSFT" or keyword == "AAPL":
        res = requests.get(f'https://api.polygon.io/v2/aggs/ticker/{keyword}/range/1/day/{prevDate}/{currDate}?adjusted=true&sort=desc&apiKey=39ngr6sQIbZKPDfWGh4yMh5dLzpL9wzf').json()

        stock_60 = pd.DataFrame(list(zip([datetime.fromtimestamp(agg['t']/1000).strftime('%Y-%m-%d') for agg in res['results']][:60], [agg['c'] for agg in res['results']][:60], [agg['o'] for agg in res['results']][:60])), columns = ['Date', 'Close', 'Open'])

        return stock_60

    if keyword == "NIFTY":
        # print(prevDate, currDate)
        data = get_history(symbol=keyword, start=prevDate, end=currDate, index=True)
        # print(data)
        a = data.index.copy()
        data = data.reset_index()
        data['Date'] = a
        data['Date'] = data['Date'].apply(correct_date_fmt)
        # print(data)
        return data[['Date','Close','Open']][:60]
        
    if keyword == "DJIA":
        dow = yf.Ticker("^DJI")
        res = dow.history(period='1d', start=str(prevDate), end=str(currDate))

        a = res.index.copy()
        res = res.reset_index()
        res['Date'] = a
        res['Date'] = res['Date'].apply(correct_date_fmt)
        
        return res[['Date', 'Close', 'Open']][:60]


def stock_60(stockArr):
    stock_60 = [agg['c'] for agg in stockArr['results']]
    return stock_60[:60]

def getRequiredScaler(keyword):
    if keyword == "AAPL":
        scaler = joblib.load('scalers/apple_scaler.save')
        model = MODEL_APPLE
        return {scaler: scaler, model: model}

    elif keyword == "MSFT":
        scaler = joblib.load('scalers/microsoft_scaler.save')
        model = MODEL_MICROSOFT
        return {scaler: scaler, model: model}

    elif keyword == "DJIA":
        scaler = joblib.load('scalers/dow_scaler.save')
        model = MODEL_DOW
        return {scaler: scaler, model: model}

    elif keyword == "NIFTY":
        scaler = joblib.load('scalers/nifty_scaler.save')
        model = MODEL_NIFTY
        return {scaler: scaler, model: model}
        


        
