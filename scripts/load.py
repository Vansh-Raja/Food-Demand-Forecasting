from pymongo import MongoClient
from sqlalchemy import create_engine
import pandas as pd
import json

with open('params.json','r') as f:
    di=json.load(f)
    db_type=di['db_type']
    db_name=di['db_name']
    table_name=di['db_url']

processed_data_path = 'D:/Study/INTERNSHIP/FINALDUPLICATE/data/processed/preprocessed_data.csv'
df=pd.read_csv(processed_data_path)
sqlite_db_path = f"D:/Study/INTERNSHIP/FINALDUPLICATE/external/{db_name}.db"

if db_type == 'mongo':
    try:
        host,port = table_name.split(':')
    except:
        host,port = 'localhost', 27017
    mongo_client = MongoClient(host, int(port))
    mongo_db = mongo_client[db_name]
    mongo_collection = mongo_db['selected_data']
    mongo_collection.insert_many(df.to_dict(orient='records'))

elif db_type == 'sqlite':
    engine = create_engine(f'sqlite:///{sqlite_db_path}')
    df.to_sql('selected_data', engine, index=False, if_exists='replace')



