import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

df = pd.read_csv('CR.csv')
print("reviews")
print(df)

column_name = 'REVIEWS'

nltk.download('vader_lexicon')
ays = SentimentIntensityAnalyzer()

for sentence in df[column_name]:
    sentiment_value = ays.polarity_scores(sentence)
    
    
    if sentiment_value['compound'] >= 0:
        sentiment = 'positive'
    else:
        sentiment = 'negative'
    
    
    print(f"Sentence: {sentence}\nSentiment: {sentiment}\n")

