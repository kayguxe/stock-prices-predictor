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
import cgi
import csv
import re
import urllib
from flask import Flask, request, render_template
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
    	count1, count2, counter_positive, counter_negative, counter_subjective, counter_objective = 0, 0, 0, 0, 0, 0
    	tweets_not_subjective = []
    	tweets_subjective = []
    	tweets_positive = []
    	tweets_negative = []
    	overall_sentiment = ""
    	overall_subjectivity = ""
    	sentiment_array = []
    	subjectivity_array = []
    	for tweets in list_of_tweets:
            # print(tweets.text)
            # print(tweets.created_at)
        	sent = TextBlob(tweets.text)
            # print(sent.sentiment)
        	if sent.sentiment.subjectivity < 0.5 and counter_objective <= 5:
        		counter_objective += 1
        		tweets_not_subjective.append(tweets)
        	elif sent.sentiment.subjectivity > 0.5 and counter_subjective <= 5:
        		tweets_subjective.append(tweets)
        		counter_subjective += 1          	
        	if sent.sentiment.polarity > 0.0 and counter_positive <= 5:
        		tweets_positive.append(tweets)
        		counter_positive += 1            	
        	elif sent.sentiment.polarity < 0.0 and counter_positive <= 5:
        		tweets_negative.append(tweets)
        		counter_negative += 1
        	count1 += sent.sentiment.subjectivity
        	count2 += sent.sentiment.polarity

        	average2 = count2 / 100.0
        	if average2 > 0:
        		overall_sentiment = "negative "
        	elif average2 <0:
        		overall_sentiment = "positive"
        	elif average2 == 0:
        		overall_sentiment = "neutral"
        	return overall_sentiment
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

# Check if the stock sentiment is positve

# Check if we got te historical data


# We have our file so we create the neural net and get the prediction
#print(stock_sentiment(query, num_tweets=100))

# We are done so we delete the csv file

app = Flask(__name__, static_url_path='/static')

@app.route("/", methods=['GET', 'POST'])

def send():
	if request.method == 'POST':
		query = request.form.get('q')
		query = query.upper()
		if not get_historical(query):
			print('<h1> Google returned a 404, please re-run the script and</h1>')
			print('enter a valid stock quote from NASDAQ')
			sys.exit()
    		
		result = stock_sentiment(query, num_tweets=1000)
		return render_template('results.html',stock_name = query, query = result, stock=stock_prediction())
		os.remove(FILE_NAME)
	return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)
