# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:56:13 2020

@author: SACHIN KESHAV
"""


import Fake_score
import objectivity_score
import bias_rank
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import string 
from collections import defaultdict
import nltk
from pattern.en import sentiment
from pattern.en.wordlist import PROFANITY
import numpy as np

    


class News_Credibility:
    def __init__(self):
        self.fake_score_object = Fake_score.FakeScore()
        self.bias_score_object = bias_rank.Bias_Score()
        self.rank_score_object = bias_rank.Ranks()
        self.obj_score_object = objectivity_score.Objective_score()
        self.final_coeff = {
                                'obj_score_coeff' : 0.3,
                                'fake_score_coeff' : 0.35,
                                'bias_score_coeff' : 0.25,
                                'google_rank_coeff' : 0.1,
                                'alexa_rank_coeff': -1*0.000001
                            }
        
        # giving a default values to all the scores
        self.fake_score = 0.5
        self.obj_score = 0.5
        self.google_score = 0.5
        self.alexa_score = 0.5
        self.bias_score = 0.5
    
    def scores(self,headline,text,url,source):

        self.fake_score = self.fake_score_object.fake_predictor(text)
        self.bias_score = self.bias_score_object.bias_score(source)
        self.google_score, self.alexa_score = self.rank_score_object.url_rank(url)
        self.obj_score = self.obj_score_object.predictions(headline, text)
        #print(self.obj_score, self.fake_score[0,0], self.bias_score, self.google_score, self.alexa_score)
        final_score = self.fake_score[0,0]*self.final_coeff['fake_score_coeff'] \
            + self.bias_score*self.final_coeff['bias_score_coeff'] \
                + self.google_score*self.final_coeff['google_rank_coeff'] \
                    +self.alexa_score * self.final_coeff['alexa_rank_coeff'] \
                        + self.obj_score[0]*self.final_coeff['obj_score_coeff']            
        #print(final_score)
        return final_score
        
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
        
if __name__ == '__main__':
    data = pd.read_csv('output1000.csv')
    headline = data['Title'][0]
    text = data['Content'][0]
    source = data['Source'][0]
    url = data['URL'][0]
    credibility = News_Credibility()
    #print(url,source)
    print(credibility.scores(headline, text, url, source))
    