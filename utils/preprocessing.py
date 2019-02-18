import pandas as pd 
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *

stemmer = PorterStemmer()
other_stopwords = ['also', 'another', 'came', 'come', 'could ', 'even', 'furthermore', 'get',
                  'got', 'hi', 'however', 'indeed', 'like', 'made', 'many', 'might', 'moreover',
                  'much', 'must', 'my', 'now of', 'said', 'see', 'since', 'still', 'take',
                  'therefore', 'thus', 'way', 'would', 'null', 'url']
stopwords_all = other_stopwords + stopwords.words('english')
stopwords_regex = r"\b({})\b".format('|'.join(map(re.escape, stopwords_all)))

def clean(df):
    # Basic cleaning
    df["clean_tweets"] = df["tweet"].str.lower()
    
    # Removing urls (those of external datasets)
    df['clean_tweets'] = df["clean_tweets"].str.replace(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "")
    df["clean_tweets"] = df["clean_tweets"].str.replace(stopwords_regex, "")
    
    # Removing usernames
    df["clean_tweets"] = df["clean_tweets"].str.replace(r"@(\w+)","")
    
    df["clean_tweets"] = df["clean_tweets"].str.replace(r"[^a-zA-Z#]"," ")
    df["clean_tweets"] = df["clean_tweets"].str.replace(r"(\b\S{1,2}\b)", "")
    df["clean_tweets"] = df["clean_tweets"].str.replace("#","")
    df["clean_tweets"] = df["clean_tweets"].str.replace(r"\s\s+"," ")
    
    # Tokenization
    #df["tokens"] = df["clean_tweets"].str.split()
    #df["tokens"] = df["tokens"].apply(lambda x: [stemmer.stem(i) for i in x])
    #df["clean_tweets"] = df["tokens"].apply(lambda x: " ".join(x))
    
    # droping empty rows
    df = df[df["clean_tweets"].str.len()>0]