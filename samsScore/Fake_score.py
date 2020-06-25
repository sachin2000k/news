# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 04:52:51 2020

@author: SACHIN KESHAV
"""
import warnings
warnings.filterwarnings("ignore")
import nltk
import pandas as pd
import re
from tensorflow.keras.models import load_model, model_from_json
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from gensim.models.phrases import  Phraser

class FakeScore:
    def __init__(self):
        #model/xgb_pipeline.pkl
        with open('model/fake_tokenizer.pickle', 'rb') as handle:
            self.tokenzs = pickle.load(handle)
        with open('model/model_in_json2.json','r') as f:
            self.model_json = json.load(f)
            #print(self.model_json)
            
        self.l_model = model_from_json(self.model_json)
        self.l_model.load_weights('model/model_weights2.h5')
        self.bigrams = Phraser.load("model/bigrams")
        self.trigrams = Phraser.load("model/trigrams")
        self.fake_score = 0
    
    def clean_str(self,string):
        neu = " neutral "
        #print(type(string))
        if type(string) != type(neu):
            string = neu
          
           
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
    
    def preprocessing(self,text):
        
        #data['text'] = data['Content'].apply(lambda x: clean_str(str(x)))
        text = self.clean_str(text)
        data = [[text]]
        data = pd.DataFrame(data, columns = ['text'])
        X2 = []
        maxlen = 700
        stop_words = set(nltk.corpus.stopwords.words("english"))
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        for par in data["text"].values:
            tmp = []
            sentences = nltk.sent_tokenize(par)
            for sent in sentences:
                sent = sent.lower()
                tokens = tokenizer.tokenize(sent)
                tokens = self.trigrams[self.bigrams[tokens]]
                filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
                tmp.extend(filtered_words)
            X2.append(tmp)
          
        X2 = self.tokenzs.texts_to_sequences(X2)
        X2 = pad_sequences(X2, maxlen=700)
        return X2
    
    def fake_predictor(self, text):
        X_test = self.preprocessing(text)
        fake_score = self.l_model.predict(X_test)
        #print(fake_score)
        return fake_score
        
    