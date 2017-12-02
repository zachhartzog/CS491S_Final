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
