import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import sys
import tweepy
import requests
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from textblob import TextBlob
from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def process_form():
    if request.method == 'POST':
        form_input = request.form['name']
        return render_template('index.html',name=form_input)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

consumer_key = '6n6m3MW5DKWl6rqtqKKgo3LaK'
consumer_secret = '6fmajLtZ6U0HVqdwfRyQMS2fD0pMHGTpzFO3cN9uBVjNIocdwd'
access_token = '726689466-iFTLEKJVqMdrXw8R6xAqJ2CLgAe6r26cV6PJjRQJ'
access_token_secret = '7by4Ts13uVeTV5ba5Bq5geg4MwfwVX3oaekBE6iRUsRQ7'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
user = tweepy.API(auth)

# Where the csv file will live
FILE_NAME = 'stock.csv'


def stock_sentiment(quote, num_tweets):
    # Checks if the sentiment for our quote is
    list_of_tweets = user.search(quote, count=num_tweets)
    positive, null = 0, 0

    for tweet in list_of_tweets:
        blob = TextBlob(tweet.text).sentiment
        if blob.subjectivity == 0:
            null += 1
            next
        if blob.polarity > 0:
            positive += 1
            print("positive")


    if positive > ((num_tweets - null)/2):
        return True


def get_historical(quote):
    url = 'http://www.google.com/finance/historical?q=NASDAQ%3A'+quote+'&output=csv'
    r = requests.get(url, stream=True)

    if r.status_code != 400:
        with open(FILE_NAME, 'wb') as f:
            for chunk in r:
                f.write(chunk)

        return True


def stock_prediction():

    dataset = []

    with open(FILE_NAME) as f:
        for n, line in enumerate(f):
            if n != 0:
                dataset.append(float(line.split(',')[1]))

    dataset = np.array(dataset)

    # Create dataset matrix (X=t and Y=t+1)
    def create_dataset(dataset):
        dataX = [dataset[n+1] for n in range(len(dataset)-2)]
        return np.array(dataX), dataset[2:]
        
    trainX, trainY = create_dataset(dataset)

    # Create and fit Multilinear Perceptron model
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

    prediction = model.predict(np.array([dataset[0]]))
    result = 'The price will move from %s to %s' % (dataset[0], prediction[0][0])

    return result

    
# Ask user for a stock quote
stock_quote = input('Enter a stock quote from NASDAQ (e.j: AAPL, FB, GOOGL): ').upper()

# Check if the stock sentiment is positve
if not stock_sentiment(stock_quote, num_tweets=100):
    print('This stock has bad sentiment, please re-run the script')

# Check if we got te historical data
if not get_historical(stock_quote):
    print('Google returned a 404, please re-run the script and')
    print('enter a valid stock quote from NASDAQ')
    sys.exit()

# We have our file so we create the neural net and get the prediction
print(stock_prediction())

# We are done so we delete the csv file
os.remove(FILE_NAME)
