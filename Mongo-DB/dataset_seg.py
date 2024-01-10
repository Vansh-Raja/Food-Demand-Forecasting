import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from pymongo import MongoClient

csv_file_path = 'train1.csv'
df = pd.read_csv(csv_file_path)

sqlite_db_path = r"C:\Users\ayush\pythonProject2023\my_database.db"

# SQLite Connection 
if not os.path.exists(sqlite_db_path):
    os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)  
    create_db_engine = create_engine(f'sqlite:///{sqlite_db_path}')
    create_db_conn = create_db_engine.connect() 
    create_db_conn.execute('''
        CREATE TABLE selected_data (
            id INT,
            week INT,
            center_id INT,
            meal_id INT,
            num_orders INT
        )
    ''')
    create_db_conn.close()  
    print('Table "selected_data" created successfully')

sql_engine = create_engine(f'sqlite:///{sqlite_db_path}')  # SQLite database

# try-except block
try:
    sql_query = 'SELECT * FROM selected_data'
    sql_data = pd.read_sql_query(sql_query, sql_engine)
    print('Selected Columns from SQLite:')
    print(sql_data)
except OperationalError as e:
    print(f"Error: {e}")

#mongoDB
#PRINT1
csv_file_path = 'C:\\Users\\ayush\\pythonProject2023\\meal_info.csv' 
df = pd.read_csv(csv_file_path)
mongo_client = MongoClient('localhost', 27017)
mongo_db = mongo_client['my_mongodb_database1']
mongo_collection = mongo_db['selected_data']
selected_columns_mongo = df[['meal_id','category','cuisine']].to_dict(orient='records')
mongo_collection.insert_many(selected_columns_mongo)
mongo_query = {}
mongo_projection = {'meal_id': 1, 'category': 1, 'cuisine': 1, '_id': 0}  # Specify columns to retrieve
mongo_data = mongo_collection.find(mongo_query, mongo_projection)
mongo_data = pd.DataFrame(list(mongo_data))
print('\nSelected Columns from MongoDB:')
print(mongo_data)


#PRINT2
csv_file_path = 'C:\\Users\\ayush\\pythonProject2023\\fulfilment_center_info.csv' 
df = pd.read_csv(csv_file_path)
mongo_client = MongoClient('localhost', 27017)
mongo_db = mongo_client['my_mongodb_database2']
mongo_collection = mongo_db['selected_data']
selected_columns_mongo = df[['center_id','city_code','region_code','center_type','op_area']].to_dict(orient='records')
mongo_collection.insert_many(selected_columns_mongo)
mongo_query = {}
mongo_projection = {'center_id':1,'city_code':1,'region_code':1,'center_type':1,'op_area':1,'_id': 0} 
mongo_data = mongo_collection.find(mongo_query, mongo_projection)
mongo_data = pd.DataFrame(list(mongo_data))
print('\nSelected Columns from MongoDB:')
print(mongo_data)
