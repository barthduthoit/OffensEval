import pandas as pd 
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
from symspellpy.symspellpy import SymSpell, Verbosity
import json

with open('data/json/contractions.json') as f:
    contractions_dict = json.load(f)

other_stopwords = ['also', 'another', 'came', 'come', 'could ', 'even', 'furthermore', 'get',
                  'got', 'hi', 'however', 'indeed', 'like', 'made', 'many', 'might', 'moreover',
                  'much', 'must', 'my', 'now of', 'said', 'see', 'since', 'still', 'take',
                  'therefore', 'thus', 'way', 'would', 'null', 'url']
stopwords_all = other_stopwords + stopwords.words('english')
stopwords_regex = r"\b({})\b".format('|'.join(map(re.escape, stopwords_all)))

stemmer = PorterStemmer()

max_edit_distance_dictionary = 2
prefix_length = 7
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
dictionary_path = "frequency_dictionary_en_82_765.txt"
term_index = 0  # column of the term in the dictionary text file
count_index = 1  # column of the term frequency in the dictionary text file
sym_spell.load_dictionary(dictionary_path, term_index, count_index)
max_edit_distance_lookup = 2
suggestion_verbosity = Verbosity.CLOSEST


def contractions_cleaning(sentence):
    if sentence != None:
        return ' '.join([contractions_dict.get(word, word) for word in sentence.split()])
    else:
        return sentence

def spell_checking(word):
    try:
        return sym_spell.lookup(word, suggestion_verbosity, max_edit_distance_lookup)[0].term
    except IndexError:
        return word




def clean(df):
    # Basic cleaning
    df["clean_tweets"] = df["tweet"].str.lower()
    
    # Removing urls (those of external datasets)
    df['clean_tweets'] = df["clean_tweets"].str.replace(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "")
    
    # Decontracting apostrophes
    df["clean_tweets"] = df["clean_tweets"].apply(contractions_cleaning)
    
    # Removing stop words
    df["clean_tweets"] = df["clean_tweets"].str.replace(stopwords_regex, "")
    
    # Removing usernames
    df["clean_tweets"] = df["clean_tweets"].str.replace(r"@(\w+)","")
    
    df["clean_tweets"] = df["clean_tweets"].str.replace(r"[^a-zA-Z#]"," ")
    
    # Removinv multi-spaces
    df["clean_tweets"] = df["clean_tweets"].str.replace(r"(\b\S{1,2}\b)", "")
    df["clean_tweets"] = df["clean_tweets"].str.replace("#","")
    df["clean_tweets"] = df["clean_tweets"].str.replace(r"\s\s+"," ")
    
    # Tokenization
    df["tokens"] = df["clean_tweets"].str.split()
    
    # Spelling
    df["tokens"] = df["tokens"].apply(lambda x: [spell_checking(i) for i in x])
    
    
    df["tokens"] = df["tokens"].apply(lambda x: [stemmer.stem(i) for i  in x])
    df["clean_tweets"] = df["tokens"].apply(lambda x: " ".join(x))
    
    # droping empty rows
    df = df[df["clean_tweets"].str.len()>0]