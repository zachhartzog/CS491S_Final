import re
import vaderSentiment as vader

analyzer = vader.SentimentIntensityAnalyzer()

def getSentiment(tweet):
    pos = analyzer.polarity_scores(tweet)['pos']
    neu = analyzer.polarity_scores(tweet)['neu']
    neg = analyzer.polarity_scores(tweet)['neg']
    if pos > neg: return pos
    if pos < neg: return (neg*(-1.0))
    if pos == neg: return 0.0
