from pymongo import MongoClient
import pandas as pd

csv_file_path = 'C:\\Users\\ayush\\pythonProject2023\\meal_info.csv' 
df = pd.read_csv(csv_file_path)

mongo_client = MongoClient('localhost', 27017)

mongo_db = mongo_client['my_mongodb_database1']

mongo_collection = mongo_db['selected_data']

selected_columns_mongo = df[['meal_id','category','cuisine']].to_dict(orient='records')
mongo_collection.insert_many(selected_columns_mongo)

mongo_query = {}
mongo_projection = {'meal_id': 1, 'category': 1, 'cuisine': 1, '_id': 0}  
mongo_data = mongo_collection.find(mongo_query, mongo_projection)
mongo_data = pd.DataFrame(list(mongo_data))

print('\nSelected Columns from MongoDB:')
print(mongo_data)
