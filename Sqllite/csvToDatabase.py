import pymongo
import csv
from sqlalchemy import create_engine as ce
from sqlalchemy import text
import pandas as pd
import json

#will create a sqllite db if not created, else will edit the existing db
def enginesql():
    db_name= input("What is your sql db name? ")
    return ce('sqlite:///'+ db_name +'.db')

#creates a database in the defined mongo collection
def create_collection_mongo():
    db = client["kunal"]
    collection = db["blrdataset"]
    return collection

#inerts the columns of the csv file into the sqllite db
def csv_to_sql(csv_name, new_table_name, cols):
    data = pd.read_csv(csv_name)
    data = data.drop(columns = cols, axis =1)
    data.to_sql(new_table_name, engine, index=False, if_exists='replace')

#inserts the defined columns into the mongo db
def csv_to_mongo(csv_name, cols_mongo, collection):
    col_sno = ["s.no"]
    cols_mongo = col_sno + cols_mongo
    
    with open(csv_name, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        rows_to_insert = [{key: row[key] for key in cols_mongo if key in row} for row in csv_reader]
        if rows_to_insert:
            collection.insert_many(rows_to_insert)

if __name__ == "__main__":
    #start databases
    client = pymongo.MongoClient("mongodb://localhost:27017")
    engine = enginesql()
    conn = engine.connect()

    #edit the create_collection function in sync with your mongodb server 
    collection = create_collection_mongo()

    # make a list of the columns you need to insert into mongodb
    cols_mongo = ["week","center_id","meal_id", "num_orders"] 

    csv_name= input("Enter CSV File name(should be in the same directory): ")
    new_sql_table_name= input("Enter the Table name to be created in sqllite db: ")
    csv_to_sql(csv_name, new_sql_table_name, cols_mongo)
    csv_to_mongo(csv_name, cols_mongo, collection)
    