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
import time
from s3 import s3_manager
from objectivity_score import HeadlineBodyFeaturesExtractor, Text_Stats, Punct_Stats, ItemSelector

class News_Credibility:
    def __init__(self):
        self.manager = s3_manager()
        self.manager.verify_bucket()

        #Downloading model for fake score
        self.manager.download_fakeTokenizer()
        self.manager.download_json_model()
        self.manager.download_lstm()
        self.manager.download_tri_bi()
        
        #Downloading model for objective score
        self.manager.download_pipeline()

        #Downloading data for bias_rank
        self.manager.download_data()
        
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

