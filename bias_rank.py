# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 05:18:37 2020

@author: SACHIN KESHAV
"""
import pandas as pd

class Bias_Score:
    
    def __init__(self):
        self.domain_info = pd.read_csv('Domain_info.csv')
        self.tag_score = {' unreliable' : 0,
                             'Conspiracy' : 0.1,
                             'Fake' : 0,
                             'Political' : 0.2,
                             'bias' : 0.2,
                             'blog': 0.3,
                             'clickbait' : 0.1,
                             'conspiracy' : 0.1,
                             'fake' : 0,
                             'fake ' : 0 ,
                             'fake news' : 0,
                             'hate' : 0.3,
                             'imposter site' : 0.2,
                             'junksci' : 0.5,
                             'least_biased' : 1,
                             'left' : 0.5,
                             'left_center' : 0.6,
                             'parody site' : 0.1,
                             'political' : 0.5,
                             'pro_science' : 0.9,
                             'pseudoscience' : 0.8,
                             'questionable' : 0.3,
                             'reliable': 0.9,
                             'right' : 0.5,
                             'right_center' :0.6,
                             'rumor' :0.1,
                             'rumor ':0.1,
                             'satire':0.2,
                             'satirical':0.2,
                             'some fake stories':0,
                             'state':0.7,
                             'unrealiable':0,
                             'unreliable':0,
                             'on':0 } #None    
        
        
    def bias_score(self,source):
        tag_l = self.domain_info[self.domain_info['Name'].str.find(source) == 0]['tags']
        count_None = 0
        score = 0.5
        for t in tag_l:
            #print(t[1:-1].split(", "))
            t = t[1:-1].split(', ')
            score -= 0.5
        #print(t)
        for i in range(len(t)):
            #print(t[i])
            score += self.tag_score[t[i][1:-1]]
            #print(t[i])
            if t[i][1:-1] == 'on':
                count_None += 1
        #print(score,count_None)
        score = score/(len(t)-count_None)
        if score<0:
            score= 0.0
        return score
    
class Ranks:
    def __init__(self):
        self.score_list = pd.read_csv('source_scores.csv')
    
    def trim_url(self,url):
        clean_url = url.replace("https://","")
        clean_url = clean_url.replace("www.","")
        clean_url = clean_url.split('/')[0]
        #print(clean_url)
        return clean_url
    
    def url_rank(self,url):
        url = self.trim_url(url)
        
        try:
            g_s = self.score_list[self.score_list['source_url'].str.find(url)==0]['google_pagerank'].values[0]
            g_s = float(g_s[1])
        except:
            #g_s = ss[ss['source_url'].str.find(source)==0]['google_pagerank'].values
            g_s = 0.
        try:
            c_s = self.score_list[self.score_list['source_url'].str.find(url)==0]['alexa_score'].values
            c_s = float(c_s)
        except:
            c_s = 0.
            #print(source,g_s,c_s)
        return g_s*0.1 , c_s