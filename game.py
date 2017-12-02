# General:
#import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing

from IPython.display import display
import re
import csv

import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# We import our access keys:
from credentials import *    # This will allow us to use the keys as variables

'''
# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api
'''
#CHANGE THIS
#=================================================================
# We create an extractor object:
'''
extractor = twitter_setup()

# We create a tweet list as follows:
tweets = extractor.search(q="Alabama Football", lang="en", count=20)
print("Number of tweets extracted: {}.\n".format(len(tweets)))

# We print the most recent 5 tweets:
print("recent tweets:\n")
for tweet in tweets:
    print(tweet.text)
    print()
'''
#=================================================================

with open('2015_lsu_alabama.csv') as myFile:
    reader = csv.reader(myFile)
    data = pd.DataFrame(data=[row[0] for row in reader], columns=['Tweets'])

display(data.head(20))

print "Getting Analysis..."
sid = SentimentIntensityAnalyzer()
for tweet in data['Tweets']:
   print(tweet)

test_data = "Manchurian was hot and spicy"
test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}

print (classifier.classify(test_data_features))


data['NLTK'] = np.array([ max(sid.polarity_scores(tweet), key=lambda x: x[1]) for tweet in data['Tweets'] ])

display(data.head(30))
'''
with open('2015_lsu_alabamaDATUM.csv', 'w') as csvfile:
    fieldnames = ['text', 'analysis']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    i = 0
    for tweet in data['Tweets']:
        print i + "\t" + tweet + "\n"
        writer.writerow({'text': tweet, 'analysis': datum_box.twitter_sentiment_analysis(tweet).encode('utf-8')})
        i++
'''
#for i in range(30):
    #print datum_box.twitter_sentiment_analysis(data['Tweets'][i]).encode('utf-8')
#data['DATUM'] = np.array([ datum_box.twitter_sentiment_analysis(tweet).encode('utf-8') for tweet in data['Tweets'] ])

# We display the updated dataframe with the new column:
display(data.head(30))

pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['NLTK'][index] == 'pos']
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['NLTK'][index] == 'neu']
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['NLTK'][index] == 'neg']

# We print percentages:
print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))
