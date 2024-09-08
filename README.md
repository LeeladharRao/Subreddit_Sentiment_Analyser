# LSF Subreddit Sentiment Analyser

This is python program that does sentimental analysis of all daily posts of Sub-Reddit page LSF. 
Written in python, uses nltk and Disitilibert for sentimental analysis
which configyred to execute everyday with Github Actions.

## Last Run Results

Last Executed on <!-- date_value starts -->
2024/09/07
<!-- date_value ends -->

According to DistilBERT, yesterday's posts on LSF were <!-- distilibert_per starts -->54<!-- distilibert_per ends -->% <!-- distilibert_value starts -->Positive<!-- distilibert_value ends -->

According to NLTK, yesterday's posts on LSF were <!-- nltk_per starts -->91<!-- nltk_per ends -->% <!-- nltk_value starts -->Negative<!-- nltk_value ends -->

## Tech

I have used the following libraries for this script:

- `PyTorch` - Open Source Machine Learning framework!
- `NLTK` - built for working with NLP
- `DistilBERT` - masked language modelling mostly used for text and speach
- `Reddit API` - set of APIs that got rate limited to hell..
- `PRAW` - package that allows access to Reddit's API.
- `Transformers` - provides thousands of pretrained models
- `Pandas` - duh
- `Hub` - best Machine Learning and AI Community  
- `Github Actions` - env to execute pretty much everything using .yaml

## Get started

Clone and install.

```sh
git clone git@github.com:LeeladharRao/Subreddit_Sentiment_Analyser
cd Subreddit_Sentiment_Analyser
pip install -r requirements.txt
```

Run the python script

```sh
py main.py
```

## Sentiment Analysis

Sentiment analysis is the process of analyzing digital text to determine if the emotional tone of the message is positive, negative, or neutral. 
Today, companies have large volumes of text data like emails, customer support chat transcripts, social media comments, and reviews.
Sentiment analysis tools can scan this text to automatically determine the author’s attitude towards a topic. Companies use the insights from sentiment analysis to improve customer service and increase brand reputation. 

## Sources

- Sentiment Analysis - [What is this? and Why?](https://monkeylearn.com/sentiment-analysis/)
- Create Reddit API app - [Reddit Dev App](https://www.reddit.com/prefs/apps)
- Reddit API Documentaion - [Docs](https://www.reddit.com/dev/api/)
- PRAW Documentation - [Docs](https://praw.readthedocs.io/en/stable/) 
- DistiBERT Documentaion - [Docs](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)


**Free Software, Hell Yeah!**
