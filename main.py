import os
import praw
from datetime import date, timedelta, datetime
from pytz import timezone 
import pandas as pd 
import nltk 
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from transformers import pipeline
import torch
import pathlib
import re


root = pathlib.Path(__file__).parent.resolve()
logger_path = root / 'log.txt'
readme_path = root / 'README.md'


def RedditAuth():
    reddit = praw.Reddit(
        client_id = os.environ['CLIENT_ID'],
        client_secret = os.environ['CLIENT_SECRET'],
        user_agent = os.environ['USER_AGENT'],
    )
    return reddit

def GetPostsData(reddit, subreddit_name, pass_date):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    
    for post in subreddit.new(limit=500):
        post_date = date.fromtimestamp(post.created_utc).strftime("%Y/%m/%d")
        if (post_date == pass_date): 
            posts_data.append({
                'title': post.title,
                'url': post.url,
                'date': post_date
            })
        elif (post_date > pass_date): 
            pass
        else: 
            break

    return posts_data

def SentimentAnalyser_NLTK(post_data):
    df = pd.DataFrame(post_data)
    sia = SentimentIntensityAnalyzer()
    results = []
    for title in df['title']:
        pol_score = sia.polarity_scores(title)
        pol_score['title'] = title
        results.append(pol_score)

    df_nltk = pd.DataFrame(results)
    df_nltk = df_nltk.drop(columns=['neg', 'neu', 'pos'])
    df_nltk['SA_nltk'] = 'NEUTRAL'
    df_nltk.loc[df_nltk['compound'] > 0.2, 'SA_nltk'] = 'POSITIVE'
    df_nltk.loc[df_nltk['compound'] < 0.2, 'SA_nltk'] = 'NEGATIVE'

    return df_nltk

def SentimentAnalyser_Distilibert(post_data):
    df = pd.DataFrame(post_data)

    sentiment_pipeline = pipeline("sentiment-analysis")
    data = df['title'].apply(lambda x: sentiment_pipeline(x)[0])
    df_distilibert = pd.DataFrame.from_records(data)

    return df_distilibert

def replace_writing(content, marker, chunk, inline=False):
    r = re.compile(
        r'<!\-\- {} starts \-\->.*<!\-\- {} ends \-\->'.format(marker, marker),
        re.DOTALL,
    )
    if not inline:
        chunk = '\n{}\n'.format(chunk)
    chunk = '<!-- {} starts -->{}<!-- {} ends -->'.format(marker, chunk, marker)
    
    return r.sub(chunk, content)

def ReWriteFile(rewritten_entries, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(rewritten_entries)

def PandasDisplaySetter():
    display = pd.options.display
    display.max_columns = 1000
    display.max_rows = 2000
    display.max_colwidth = 199
    display.width = 1000

def CalculatePercentages(df):
    percentages = {}
    nltk_pos_percentage = int(df.SA_nltk.value_counts()['POSITIVE']/len(df)*100)
    if (nltk_pos_percentage > 50):
        percentages['nltk'] = {'percentage': nltk_pos_percentage, 'value': 'Positive'}
    else :
        percentages['nltk'] = {'percentage': 100-nltk_pos_percentage, 'value': 'Negative'}

    distilibert_pos_percentage = int(df.SA_distilibert.value_counts()['POSITIVE']/len(df)*100)
    if (distilibert_pos_percentage > 50):
        percentages['distilibert'] = {'percentage': distilibert_pos_percentage, 'value': 'Positive'}
    else :
        percentages['distilibert'] = {'percentage': 100-distilibert_pos_percentage, 'value': 'Negative'}

    return percentages



if __name__ == '__main__':
    subreddit_name = 'LivestreamFail'
    pass_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y/%m/%d")

    reddit = RedditAuth()
    posts = GetPostsData(reddit, subreddit_name, pass_date)

    df = pd.DataFrame(posts) 

    df_nltk = SentimentAnalyser_NLTK(posts)
    df = df.merge(df_nltk, left_on='title', right_on='title')

    df_distilibert = SentimentAnalyser_Distilibert(posts)
    df['SA_distilibert'] = df_distilibert['label']

    percentages = CalculatePercentages(df)
    PandasDisplaySetter()

    readme = open(readme_path, encoding="utf8").read()
    rewritten_entries = replace_writing(readme, 'date_value', pass_date)
    rewritten_entries = replace_writing(rewritten_entries, 'distilibert_per', percentages['distilibert']['percentage'], inline=True)
    rewritten_entries = replace_writing(rewritten_entries, 'distilibert_value', percentages['distilibert']['value'], inline=True)
    rewritten_entries = replace_writing(rewritten_entries, 'nltk_per', percentages['nltk']['percentage'], inline=True)
    rewritten_entries = replace_writing(rewritten_entries, 'nltk_value', percentages['nltk']['value'], inline=True)
    ReWriteFile(rewritten_entries, readme_path)

    logger = open(logger_path, encoding="utf8").read()
    rewritten_entries = replace_writing(logger, 'posts', df)
    rewritten_entries = replace_writing(rewritten_entries, 'SA_nltk', df.SA_nltk.value_counts())
    rewritten_entries = replace_writing(rewritten_entries, 'SA_distilibert', df.SA_distilibert.value_counts())
    ReWriteFile(rewritten_entries, logger_path)


    # print('-----------------------------------------------------')
    # print(df)
    # print('-----------------------------------------------------')
    # print(df.SA_nltk.value_counts())
    # print('-----------------------------------------------------')
    # print(df.SA_distilibert.value_counts())
    # print('-----------------------------------------------------')
