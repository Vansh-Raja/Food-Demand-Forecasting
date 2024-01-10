from pymongo import MongoClient
import pandas as pd
csv_file_path = 'C:\\Users\\ayush\\pythonProject2023\\fulfilment_center_info.csv' # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

mongo_client = MongoClient('localhost', 27017)

mongo_db = mongo_client['my_mongodb_database2']

mongo_collection = mongo_db['selected_data']

selected_columns_mongo = df[['center_id','city_code','region_code','center_type','op_area']].to_dict(orient='records')
mongo_collection.insert_many(selected_columns_mongo)

mongo_query = {}
mongo_projection = {'center_id':1,'city_code':1,'region_code':1,'center_type':1,'op_area':1,'_id': 0}  # Specify columns to retrieve
mongo_data = mongo_collection.find(mongo_query, mongo_projection)
mongo_data = pd.DataFrame(list(mongo_data))

print('\nSelected Columns from MongoDB:')
print(mongo_data)
