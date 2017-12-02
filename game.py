from pandas import Series, DataFrame # To handle data
import numpy as np      # For number computing

from IPython.display import display
import re
import csv

import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')

#from nltk.sentiment.vader import SentimentIntensityAnalyzer
import vaderSentiment as vader

datas = []
tweet = []
vs_compound = []
vs_pos = []
vs_neu = []
vs_neg = []
fav = []
rt = []
sscore = []

with open('1002.tsv') as myFile:
    reader = csv.reader(myFile, delimiter='\t')
    for row in reader:
        datas.append([row[0],row[1],row[2]])

data = DataFrame(datas, columns=['Tweets', 'Retweets','Favorites'])

display(data.head())

analyzer = vader.SentimentIntensityAnalyzer()

for i in range(0, len(data)):
    tweet.append(data['Tweets'][i])
    vs_compound.append(analyzer.polarity_scores(data['Tweets'][i])['compound'])
    vs_pos.append(analyzer.polarity_scores(data['Tweets'][i])['pos'])
    vs_neu.append(analyzer.polarity_scores(data['Tweets'][i])['neu'])
    vs_neg.append(analyzer.polarity_scores(data['Tweets'][i])['neg'])
    if vs_pos[i] > vs_neg[i]:
        sscore.append(vs_pos[i])
    if vs_pos[i] < vs_neg[i]:
        sscore.append(vs_neg[i]*(-1.0))
    if vs_pos[i] == vs_neg[i]:
        sscore.append(0.0)


data['Positive'] = np.array(vs_pos)
data['Negative'] = np.array(vs_neg)
data['Neutral'] = np.array(vs_neu)
data['Compound'] = np.array(vs_compound)
data['Sentiment Score'] = np.array(sscore)

'''
sid = SentimentIntensityAnalyzer()
for tweet in data['Tweets']:
    print tweet
    print sid.polarity_scores(tweet)

data['NLTK'] = np.array([ max(sid.polarity_scores(tweet), key=lambda x: x[1]) for tweet in data['Tweets'] ])


pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['NLTK'][index] == 'pos']
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['NLTK'][index] == 'neu']
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['NLTK'][index] == 'neg']
print pos_tweets
print neg_tweets
print neu_tweets
'''
display(data.head(30))
