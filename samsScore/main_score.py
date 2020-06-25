import requests
import json

from score import News_Credibility


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
        'size':1
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



def credibility_score(data):
    model = News_Credibility()
    result = model.score(data['title'], data['content'], data['url'], data['source'])
    print(result)
    return result

if __name__ == '__main__':

    from pyspark import SparkConf
    from pyspark import SparkContext
    conf = SparkConf().setAppName("app")
    sc = SparkContext(conf=conf)

    data = get_data_from_db()
    rdd = sc.parallelize(data,1)

    result = rdd.foreach(credibility_score)




