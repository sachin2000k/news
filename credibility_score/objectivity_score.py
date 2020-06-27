import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import string
from random import shuffle
from textblob import TextBlob
import nltk
import re

class Objective_score:
    def __init__(self):
        with open('objectScore/dict_transformer_hp.pickle', 'rb') as handle:
            self.dict_transformer_hp_ = pickle.load(handle)
            print(self.dict_transformer_hp_)
        with open('objectScore/dict_transformer_bp.pickle', 'rb') as handle:
            self.dict_transformer_bp_ = pickle.load(handle)
        with open('objectScore/dict_transformer_ht.pickle', 'rb') as handle:
            self.dict_transformer_ht_ = pickle.load(handle)
        with open('objectScore/dict_transformer_bt.pickle', 'rb') as handle:
            self.dict_transformer_bt_ = pickle.load(handle)
        with open('objectScore/count_vectorizer_h.pickle', 'rb') as handle:
            self.count_vectorizer_h_ = pickle.load(handle)
        with open('objectScore/count_vectorizer_b.pickle', 'rb') as handle:
            self.count_vectorizer_b_ = pickle.load(handle)
        with open('objectScore/xgb_obj_model.pkl', 'rb') as file:  
            self.xgb_model = pickle.load(file)

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
        #string = re.sub(r"\?", "", string)
        string = re.sub(r"'", "", string)
        string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
        string = re.sub(r"[0-9]\w+|[0-9]","", string)
        string = re.sub(r"\s{2,}", " ", string)
        #string = ' '.join(Word(word).lemmatize() for word in string.split() if word not in STOPWORDS) # delete stopwors from text

        return string.strip()
    

    def HeadlineBodyFeaturesExtractor(self,posts):
        #print("in HB features")
        features = np.recarray(shape = (len(posts),), dtype = [('headline', object), ('article_body', object), ('headline_pos', object), ('body_pos', object)])
        #print(posts)
        for i, post in enumerate(posts):
            if i%100 == 0:
                print(i)
            #print(post)
            headline, article = post[:2]
            features[i]['headline'] = headline
            features[i]['article_body'] = article
            tok_headline = nltk.word_tokenize(headline)
            features[i]['headline_pos'] = (' ').join([x[1] for x in nltk.pos_tag(tok_headline)])
            tok_article = nltk.word_tokenize(article)
            features[i]['body_pos'] = (' ').join([x[1] for x in nltk.pos_tag(tok_article)])
        #print("ending HB features")
        #print(features)
        return features

    def Punct_Stats(self,text_fields):
        #print("entring Punct Stats")
        punct_stats = []
        punctuations = list(string.punctuation)
        additional_punct = ['``', '--', '\'\'']
        punctuations.extend(additional_punct)
        
        #print(text_fields)
        for field in text_fields:
            if field == None:
                field = "Neutral " 
            puncts = defaultdict(int)
            for ch in field:
                if ch in punctuations:
                    puncts[ch] += 1
            punct_stats.append(puncts)
        #print("ending Punct_stats")
        return punct_stats

    def Text_Stats(self,text_fields):
        #print("entering Text Stats")
        stats = []
        punctuation = string.punctuation
        abvs = ['CNN', 'FBI', 'ABC', 'MSNBC', 'GOP', 'U.S.', 'US', 'ISIS', 'DNC', 'TV', 'CIA', 'I', 'AP', 'PM', 'AM', 'EU', 'USA', 'UK', 'UN', 'CEO', 'NASA', 'LGBT', 'LGBTQ', 'NAFTA', 'ACLU']
        PROFANITY = ['anus', 'arse', 'arsehole', 'ass', 'ass-hat', 'ass-jabber', 'ass-pirate', 'assbag', 'assbandit', 'assbanger', 'assbite', 'assclown', 'asscock', 'asscracker', 'asses', 'assface', 'assfuck', 'assfucker', 'assgoblin', 'asshat', 'asshead', 'asshole', 'asshopper', 'assjacker', 'asslick', 'asslicker', 'assmonkey', 'assmunch', 'assmuncher', 'assnigger', 'asspirate', 'assshit', 'assshole', 'asssucker', 'asswad', 'asswipe', 'balls', 'bampot', 'bastard', 'beaner', 'bint', 'bitch', 'bitchass', 'bitches', 'bitchtits', 'bitchy', 'bloody', 'blowjob', 'blowjob', 'bollocks', 'bollox', 'boner', 'brotherfucker', 'bugger', 'bullshit', 'bumblefuck', 'butt plug', 'butt-pirate', 'buttfucka', 'buttfucker', 'camel toe', 'carpetmuncher', 'chinc', 'chink', 'choad', 'chode', 'clit', 'clitface', 'clitfuck', 'clusterfuck', 'cock', 'cockass', 'cockbite', 'cockburger', 'cockface', 'cockfucker', 'cockhead', 'cockjockey', 'cockknoker', 'cockmaster', 'cockmongler', 'cockmongruel', 'cockmonkey', 'cockmuncher', 'cocknose', 'cocknugget', 'cockshit', 'cocksmith', 'cocksmoke', 'cocksmoker', 'cocksniffer', 'cocksucker', 'cockwaffle', 'coochie', 'coochy', 'coon', 'cooter', 'cracker', 'cum', 'cumbubble', 'cumdumpster', 'cumguzzler', 'cumjockey', 'cumslut', 'cumtart', 'cunnie', 'cunnilingus', 'cunt', 'cuntass', 'cuntface', 'cunthole', 'cuntlicker', 'cuntrag', 'cuntslut', 'dago', 'dammit', 'damn', 'dang', 'deggo', 'dick', 'dickbag', 'dickbeaters', 'dickface', 'dickfuck', 'dickfucker', 'dickhead', 'dickhole', 'dickjuice', 'dickmilk', 'dickmonger', 'dicks', 'dickslap', 'dicksucker', 'dicksucking', 'dickwad', 'dickweasel', 'dickweed', 'dickwod', 'dike', 'dildo', 'dipshit', 'doochbag', 'dookie', 'douche', 'douche-fag', 'douchebag', 'douchewaffle', 'dumass', 'dumb ass', 'dumbass', 'dumbfuck', 'dumbshit', 'dumshit', 'dyke', 'fag', 'fagbag', 'fagfucker', 'faggit', 'faggot', 'faggotcock', 'fagtard', 'fatass', 'fellatio', 'feltch', 'flamer', 'fool', 'frickin', 'friggin', 'f*ck', 'fuck', 'fuckass', 'fuckbag', 'fuckboy', 'fuckbrain', 'fuckbutt', 'fucked', 'fucker', 'fuckersucker', 'fuckface', 'fuckhead', 'fuckhole', 'fuckin', 'fucking', 'fucknut', 'fucknutt', 'fuckoff', 'fucks', 'fuckstick', 'fucktard', 'fucktart', 'fuckup', 'fuckwad', 'fuckwit', 'fuckwitt', 'fudgepacker', 'gay', 'gayass', 'gaybob', 'gaydo', 'gayfuck', 'gayfuckist', 'gaylord', 'gaytard', 'gaywad', 'goddamn', 'goddamnit', 'gooch', 'gook', 'gringo', 'guido', 'handjob', 'hard on', 'heeb', 'helminth', 'hell', 'ho', 'hoe', 'hoebag', 'homo', 'homodumbshit', 'honkey', 'humping', 'idiot', 'imbecile', 'jackass', 'jap', 'jerk off', 'jerk wad', 'jigaboo', 'jizz', 'jungle bunny', 'junglebunny', 'kike', 'kooch', 'kootch', 'kraut', 'kunt', 'kyke', 'lameass', 'lesbian', 'lesbo', 'lezzie', 'mcfagget', 'mick', 'midget', 'minge', 'moron', 'mothafucka', 'mothafuckin', 'motherfuck', 'motherfucker', 'motherfucking', 'muff', 'muffdiver', 'munging', 'negro', 'nigaboo', 'nigga', 'nigger', 'niggers', 'niglet', 'nutter', 'nut sack', 'nutsack', 'paki', 'panooch', 'pecker', 'peckerhead', 'penis', 'penisbanger', 'penisfucker', 'penispuffer', 'piss', 'pissed', 'pissed off', 'pissflaps', 'polesmoker', 'pollock', 'poon', 'poonani', 'poonany', 'poontang', 'porch monkey', 'porchmonkey', 'prick', 'punanny', 'punta', 'pussies', 'pussy', 'pussylicking', 'puto', 'queef', 'queer', 'queerbait', 'queerhole', 'renob', 'retard', 'rimjob', 'ruski', 'sand nigger', 'sandnigger', 'schlong', 'schmuck', 'scrote', 'scullion', 'shag', 'shit', 'shitass', 'shitbag', 'shitbagger', 'shitbrains', 'shitbreath', 'shitcanned', 'shitcunt', 'shitdick', 'shitface', 'shitfaced', 'shithead', 'shithole', 'shithouse', 'shitspitter', 'shitstain', 'shitter', 'shittiest', 'shitting', 'shitty', 'shiz', 'shiznit', 'skank', 'skeet', 'skullfuck', 'slag', 'slapper', 'slut', 'slutbag', 'slubberdegullion', 'smeg', 'snatch', 'sodding', 'sonofabitch', 'spastic', 'spic', 'spick', 'splooge', 'spook', 'sucka', 'suckass', 'sucker', 'suckers', 'tard', 'testicle', 'thundercunt', 'tit', 'titfuck', 'tits', 'tittyfuck', 'trollop', 'twat', 'twatlips', 'twats', 'twatwaffle', 'unclefucker', 'va-j-j', 'vag', 'vagina', 'vajayjay', 'vjayjay', 'wank', 'wanker', 'wankjob', 'wetback', 'whore', 'whorebag', 'whoreface', 'wop', 'wtf']
        for field in text_fields:
            field_stats = {}
            tok_text = nltk.word_tokenize(field)
            try:
                num_upper = float(len([w for w in tok_text if w.isupper() and w not in abvs]))/len(tok_text)
            except:
                num_upper = 0
            try:
                num_punct = float(len([ch for ch in field if ch in punctuations]))/len(field)
            except:
                num_punct = 0
            try:
                sent_lengths = [len(nltk.word_tokenize(s)) for s in nltk.sent_tokenize(field)]
                av_sent_len = float(sum(sent_lengths)/len(sent_lengths))
            except:
                av_sent_len = 0
            try:

                num_prof = float(len([w for w in tok_text if w.lower() in PROFANITY]))/len(tok_text)
            except:
                num_prof = 0
            
            #polarity, subjectivity = sentiment(field)
            testimonial = TextBlob(field)
            polarity = testimonial.sentiment.polarity
            subjectivity = testimonial.sentiment.subjectivity
            field_stats['all_caps'] = num_upper
            field_stats['sent_len'] = av_sent_len
            field_stats['polarity'] = polarity
            field_stats['profanity'] = num_prof
            field_stats['subjectivity'] = subjectivity
            stats.append(field_stats)
        #print("ending Text Stats")
        return stats

    def feature_generations(self,data):
        #data is in the form of list of lists
        hbf_test = self.HeadlineBodyFeaturesExtractor(data)
        headline_punct_stats = self.Punct_Stats(hbf_test['headline'])
        headline_punct_stats = self.dict_transformer_hp_.transform(headline_punct_stats)
        article_body_punct_stats = self.Punct_Stats(hbf_test['article_body'])
        article_body_punct_stats = self.dict_transformer_bp_.transform(article_body_punct_stats)

        headline_pos_cv = self.count_vectorizer_h_.transform(hbf_test['headline_pos'])
        body_pos_cv = self.count_vectorizer_b_.transform(hbf_test['body_pos'])

        text_stats_headline = self.Text_Stats(hbf_test['headline'])
        text_stats_headline = self.dict_transformer_ht_.transform(text_stats_headline)
        text_stats_body = self.Text_Stats(hbf_test['article_body'])
        text_stats_body = self.dict_transformer_bt_.transform(text_stats_body)
        hps = headline_punct_stats.todense()
        abps = article_body_punct_stats.todense()
        hpcv = headline_pos_cv.todense()
        bpcv = body_pos_cv.todense()
        tsh = text_stats_headline.todense()
        tsb = text_stats_body.todense()
        params2 = np.concatenate((hps,abps,hpcv,bpcv,tsh,tsb),axis = 1)
        #print(params2.shape)
        return params2
    
    def predictions(self,headline,text):
        headline = self.clean_str(headline)
        text = self.clean_str(text)
        data = [[headline, text]]
        features = self.feature_generations(data)
        obj_scores = self.xgb_model.predict_proba(features)[:,0]
        return obj_scores 
