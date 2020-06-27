import requests
import json
import pandas as pd

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from credibility_score import *




def get_data_from_db():
    host = 'https://search-solytics-tzxvnvrvkscgklaedz5lr3iqxu.ap-south-1.es.amazonaws.com/news_test_list/_search'
    json_body = '''{
        "query": {
                "bool": {
                    "must_not": {
                        "exists":{
                            "field":"sentiment_ML"
                            }
                        }
                    }
                }
    }'''
    headers = {
        'Content-Type': 'application/json',
    }
    params = {
        'size':15
    }
    resp = requests.get(host,params=params, headers=headers, data=json_body)
    resp_text = json.loads(resp.text)
    document_list = []

    # print(resp)
    for data in resp_text['hits']['hits']:
        content_list = {}
        content_list["id"] = data["_id"]
        content_list["content"] = data["_source"]["Content"]
        content_list["url"] = data["_source"]["URL"]
        content_list["source"] = data["_source"]["Source"]
        content_list["title"] = data["_source"]["Title"]
        document_list.append(content_list)
    return document_list


class spark_data_frame:
    def createData(self,data):
        title = []
        content=[]
        url=[]
        source=[]

        title.append(data[0]['title'])
        content.append(data[0]['content'])
        url.append(data[0]['url'])
        source.append(data[0]['source'])
        dic = {"title":title, "content":content, "url":url,"source":source}

        df_pd = pd.DataFrame(dic)
        return df_pd

    def spark_data(self):
        self.data = get_data_from_db()
        self.df_pd = self.createData(self.data)
        df = spark.createDataFrame(self.df_pd)
        return df


def credibility_score(title,content,url,source):
    manager = s3_manager()
    manager.verify_bucket()

    #Downloading model for fake score
    manager.download_fakeTokenizer()
    manager.download_json_model()
    manager.download_lstm()
    manager.download_tri_bi()

    #Downloading model for objective score
    manager.download_pipeline()

    #Downloading data for bias_rank
    manager.download_data()

    fake_score_object = FakeScore()
    bias_score_object = Bias_Score()
    rank_score_object = Ranks()
    obj_score_object = Objective_score()
    final_coeff = {
                            'obj_score_coeff' : 0.3,
                            'fake_score_coeff' : 0.35,
                            'bias_score_coeff' : 0.25,
                            'google_rank_coeff' : 0.1,
                            'alexa_rank_coeff': -1*0.000001
                        }

    # giving a default values to all the scores
    fake_score = 0.5
    obj_score = 0.5
    google_score = 0.5
    alexa_score = 0.5
    bias_score = 0.5


    fake_score = fake_score_object.fake_predictor(content)
    google_score, alexa_score = rank_score_object.url_rank(url)
    bias_score = bias_score_object.bias_score(source)


    final_score = fake_score[0,0]*final_coeff['fake_score_coeff'] \
            + bias_score*final_coeff['bias_score_coeff'] \
                + google_score*final_coeff['google_rank_coeff'] \
                    +alexa_score * final_coeff['alexa_rank_coeff']\
                        + obj_score[0]*final_coeff['obj_score_coeff']  

    return float(final_score)


#-----------------------------------------------------------

from pyspark.sql import SparkSession
spark = SparkSession.builder \
.master("local") \
.appName("Word Count") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()

obj = spark_data_frame()
df = obj.spark_data()

final_udf = udf(credibility_score,FloatType())

result = df.select("title","content","url","source",final_udf("title","content","url","source").alias('output'))
print(result.show())
