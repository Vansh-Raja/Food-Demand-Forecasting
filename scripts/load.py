from pymongo import MongoClient
from sqlalchemy import create_engine
import pandas as pd

def load(df):
    sql_df = df[['id', 'week', 'center_id', 'meal_id','checkout_price','base_price', 'emailer_for_promotion', 'homepage_featured', 'num_orders']]
    engine = create_engine(f'sqlite:///{sqlite_db_path}')
    sql_df.to_sql('selected_data', engine, index=False, if_exists='replace')
    mongo_db1 = mongo_client['my_mongodb_database1']
    mongo_collection1 = mongo_db1['selected_data']
    selected_columns_mongo1 = df[['meal_id','category','cuisine']].to_dict(orient='records')
    mongo_collection1.insert_many(selected_columns_mongo1)

    mongo_db2 = mongo_client['my_mongodb_database2']
    mongo_collection2 = mongo_db2['selected_data']
    selected_columns_mongo2 = df[['center_id','city_code','region_code','center_type','op_area']].to_dict(orient='records')
    mongo_collection2.insert_many(selected_columns_mongo2)

sqlite_db_path = 'D:/Study/INTERNSHIP/ETL1/external/preprocessed_data.db'
mongo_client = MongoClient('localhost', 27017)

processed_data_path = 'D:/Study/INTERNSHIP/ETL1/data/processed/preprocessed_data.csv'

df=pd.read_csv(processed_data_path)
load(df)
