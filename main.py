import os
import praw
from datetime import date, timedelta
import pandas as pd 
import nltk 
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from transformers import pipeline
import torch


reddit = praw.Reddit(
    client_id = os.environ["client_id"],
    client_secret = os.environ["client_secret"],
    user_agent = os.environ["user_agent"],
)

subreddit = reddit.subreddit('LivestreamFail')

titles = set()
yesterday_date = (date.today() - timedelta(days=1)).strftime("%Y/%m/%d")
for post in subreddit.new(limit=1000):
    post_date = date.fromtimestamp(post.created_utc).strftime("%Y/%m/%d")
    if (post_date == yesterday_date):
        titles.add(post.title)
    elif (post_date > yesterday_date): pass
    else : break

df = pd.DataFrame(titles) 
df.columns = ['titles']

sia = SentimentIntensityAnalyzer()
results = []
for title in titles:
    pol_score = sia.polarity_scores(title)
    pol_score['titles'] = title
    results.append(pol_score)
df_nltk = pd.DataFrame(results)
df_nltk['label'] = 'NEUTRAL'
df_nltk.loc[df_nltk['compound'] > 0.2, 'label'] = 'POSITIVE'
df_nltk.loc[df_nltk['compound'] < 0.2, 'label'] = 'NEGATIVE'

df['SA_nltk'] = df_nltk['label']


sentiment_pipeline = pipeline("sentiment-analysis")

data = df['titles'].apply(lambda x: sentiment_pipeline(x)[0])
df_distilibert = pd.DataFrame.from_records(data)
df['SA_distilibert'] = df_distilibert['label']

print("-----------------------------------------------------")
print(df)
print("-----------------------------------------------------")
print(df.SA_nltk.value_counts())
print("-----------------------------------------------------")
print(df.SA_distilibert.value_counts())
print("-----------------------------------------------------")

