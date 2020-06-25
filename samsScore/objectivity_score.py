# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 05:52:51 2020

@author: SACHIN KESHAV
"""

import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import string 
import nltk
from pattern.en import sentiment
from pattern.en.wordlist import PROFANITY
import re


class Objective_score() :
    def __init__(self):
        with open('model/xgb_pipeline.pkl','rb') as pkl:
            self.l_pipeline = pickle.load(pkl)
            
    def clean_str(self,string):
        neu = " neutral "
        #print(type(string))
        if type(string) != type(neu):
            string = neu
   
        #print(string)
        string = re.sub(r"^b", "", string)
        string = re.sub(r"\\n ", "", string)
        string = re.sub(r"\'s", "", string)
        string = re.sub(r"\'ve", "", string)
        string = re.sub(r"n\'t", "", string)
        string = re.sub(r"\'re", "", string)
        string = re.sub(r"\'d", "", string)
        string = re.sub(r"\'ll", "", string)
        string = re.sub(r",", "", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", "", string)
        string = re.sub(r"\)", "", string)
        string = re.sub(r"\?", "", string)
        string = re.sub(r"'", "", string)
        string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
        string = re.sub(r"[0-9]\w+|[0-9]","", string)
        string = re.sub(r"\s{2,}", " ", string)
        #string = ' '.join(Word(word).lemmatize() for word in string.split() if word not in STOPWORDS) # delete stopwors from text

        return string.strip()
    
    def load_data(self,headline,text):
        headline = self.clean_str(headline)
        text = self.clean_str(text)
        return headline, text
    
    def predictions(self, headline, text):
        headline, text = self.load_data(headline, text)
        test_data = [[headline,text]]
        #print(test_data)
        obj_scores = self.l_pipeline.predict_proba(test_data)[:,0]
        return obj_scores



class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        
        return data_dict[self.key]
    
class Punct_Stats(BaseEstimator, TransformerMixin):
    """Extract punctuation features from each document"""

    def fit(self, x, y=None):
        return self

    def transform(self, text_fields):
        punct_stats = []
        punctuations = list(string.punctuation)
        for field in text_fields:
            if field == None:
                field = " "
            puncts = defaultdict(int)
            for ch in field:
                if ch in punctuations:
                    puncts[ch]+=1
            punct_stats.append(puncts)
        return punct_stats
    
class Text_Stats(BaseEstimator, TransformerMixin):
    """Extract text statistics from each document"""

    def fit(self, x, y=None):
        return self

    def transform(self, text_fields):
        stats = []
       
        # abbreviations are used for not to be count in the capital letters features
        abvs = ['RBI','BCCI','CM','BJP','BSP','ICU','PPE','CNN', 'FBI', 'ABC', 'MSNBC', 'GOP', 'U.S.', 'US', 'ISIS', 'DNC', 'TV', 'CIA', 'I', 'AP', 'PM', 'AM', 'EU', 'USA', 'UK', 'UN', 'CEO', 'NASA', 'LGBT', 'LGBTQ', 'NAFTA', 'ACLU']
        for field in text_fields:
            field_stats = {}
            tok_text = nltk.word_tokenize(field)
            try:
                num_upper = float(len([w for w in tok_text if w.isupper() and w not in abvs]))/len(tok_text)
            except:
                num_upper = 0
     
            try:
                sent_lengths = [len(nltk.word_tokenize(s)) for s in nltk.sent_tokenize(field)]
                av_sent_len = float(sum(sent_lengths))/len(sent_lengths)
            except:
                av_sent_len = 0
            try:
                num_prof = float(len([w for w in tok_text if w.lower() in PROFANITY]))/len(tok_text)
            except:
                num_prof = 0

            polarity, subjectivity = sentiment(field)
            field_stats['all_caps'] = num_upper
            field_stats['sent_len'] = av_sent_len
            field_stats['polarity'] = polarity
            field_stats['subjectivity'] = subjectivity
            field_stats['profanity'] = num_prof
            stats.append(field_stats)
        return stats
    

#HYPOTHESIS: sensational news uses more pronouns, adjectives
class HeadlineBodyFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, posts):
        #punctuation = string.punctuation
        #print(len(posts))
        #print(posts)
        features = np.recarray(shape=(len(posts),), dtype=[('headline', object), ('article_body', object), ('headline_pos', object), ('body_pos', object)])
        for i, post in enumerate(posts): 
            #if i%100 == 0:
            #print(post)
            
            headline, article = post[:2]
            features['headline'][i] = headline
            features['article_body'][i] = article
            
            tok_headline = nltk.word_tokenize(headline)
            features['headline_pos'][i] = (' ').join([x[1] for x in nltk.pos_tag(tok_headline)])

            tok_article = nltk.word_tokenize(article)
            features['body_pos'][i] = (' ').join([x[1] for x in nltk.pos_tag(tok_article)])
            
            #print(features)
        return features
    
