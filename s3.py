import boto3
from os import listdir , mkdir
from os.path import isfile, join , isdir
from progress.bar import Bar
import json

'''Exception class to simplify exceptions'''
class SamsResourceError(Exception):

  def __init__(self,*args):
    if args:
        self.message = args[0]
    else:
        self.message = None

  def __str__(self):
    if self.message:
        return f'{self.message}'
    else:
        return f'Please check all resources'


class s3_manager:
    '''Need to initialize .config.json in folder'''
    def __init__(self):
        try:
            with open('.config.json','r') as config:
                val = json.loads(config.read())
                ACCESS_ID = val['aws_access_key_id']
                ACCESS_KEY = val['aws_secret_access_key']
                self.s3 = boto3.resource('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key=ACCESS_KEY)
        except:
            raise SamsResourceError('.config.json missing')
    
    '''Check model are present in bucket or not'''
    def verify_bucket(self):
        files=[]
        try:
            bucket = self.s3.Bucket('sams-models')
            for key in bucket.objects.all():
                files.append(key.key)
        except Exception as e:
            raise SamsResourceError(f"S3 Bucket error: {e}")

        if 'xgb_pipeline.pkl' not in files:
            raise SamsResourceError(f"xgb_pipeline.pkl not found in {bucket.name}")

        if 'model_weights2.h5' not in files:
            raise SamsResourceError(f"model_weights2.h5 not found in {bucket.name}")

    '''Download LSTM model as model_weights2.h5 from s3 into model folder'''
    def download_lstm(self):
        if isfile('model/model_weights2.h5'):
            print("LSTM model imported")
        else:
            bucket_name = 'sams-models'
            if not isdir('model'):
                mkdir('model')
            try:
                self.s3.meta.client.download_file(bucket_name,'model_weights2.h5','model/model_weights2.h5')
                print("LSTM Downloaded")
            except:
                raise SamsResourceError(f"LSTM model not present in {bucket_name.name}")
    
    def download_pipeline(self):
        if isfile('model/xgb_pipeline.pkl'):
            print("Pipeline Imported")
        else:
            bucket_name = 'sams-models'
            if not isdir('model'):
                mkdir('model')
            try:
                self.s3.meta.client.download_file(bucket_name,'xgb_pipeline.pkl','model/xgb_pipeline.pkl')
                print("Pipeline Downloaded")
            except:
                raise SamsResourceError(f"Pipeline is not present in {bucket_name.name}")
    
    def download_fakeTokenizer(self):
        if isfile('model/fake_tokenizer.pickle'):
            print("Fake_tokenizer is imported")
        else:
            bucket_name = 'sams-models'
            if not isdir('model'):
                mkdir('model')
            try:
                self.s3.meta.client.download_file(bucket_name,'fake_tokenizer.pickle','model/fake_tokenizer.pickle')
                print("Fake_tokenizer Downloaded")
            except:
                raise SamsResourceError(f"Fake_tokenizer is not present in {bucket_name.name}")

    def download_json_model(self):
        if isfile('model/model_in_json2.json'):
            print("Model_in_json2 is imported")
        else:
            bucket_name = 'sams-models'
            if not isdir('model'):
                mkdir('model')
            try:
                self.s3.meta.client.download_file(bucket_name,'model_in_json2.json','model/model_in_json2.json')
                print("Model_in_json2 Downloaded")
            except:
                raise SamsResourceError(f"Model_in_json2 is not present in {bucket_name.name}")
    
    def download_tri_bi(self):
        if isfile('model/bigrams'):
            print("Bigram is imported")
        else:
            bucket_name = 'sams-models'
            if not isdir('model'):
                mkdir('model')
            try:
                self.s3.meta.client.download_file(bucket_name,'bigrams','model/bigrams')
                print("Bigrams Downloaded")
            except:
                raise SamsResourceError(f"Bigrams is not present in {bucket_name.name}")
        

        if isfile('model/trigrams'):
            print("Trigram is Imported")
        else:
            bucket_name = 'sams-model'
            try:
                self.s3.meta.client.download_file(bucket_name,'trigrams','model/trigrams')
                print("Trigrams Download")
            except:
                raise SamsResourceError(f"Trigrams is not present in {bucket_name.name}")
            
