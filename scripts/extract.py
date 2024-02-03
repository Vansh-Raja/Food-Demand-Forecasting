from confluent_kafka import Consumer
import pandas as pd
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StringType, IntegerType,FloatType
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'
from pyspark.sql.functions import col, count,desc,min,mean
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import pyspark.pandas as ps
import sqlite3
import schedule
import time



def merge():
    time.sleep(12) 
    csv_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
    merged_df = pd.DataFrame()
    for csv_file in csv_files:
        file_path = os.path.join(raw_data_path, csv_file)
        df = pd.read_csv(file_path, header = None)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        os.remove(file_path)
    merged_df.to_csv(raw_data_path+'/final.csv', index=False)


#our enviroment variables
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
raw_data_path = 'D:/uni/Python/Pranav_Gryffindor_script/ETL1 (2)/ETL1/data/raw'
output_path = 'D:/uni/Python/Pranav_Gryffindor_script/ETL1 (2)/ETL1/scripts/output'
checkpoint_path= 'D:/uni/Python/Pranav_Gryffindor_script/ETL1 (2)/ETL1/scripts/checkpoint'


#building our spark session
spark = SparkSession.builder \
    .appName("KafkaConsumer") \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')

#schema of the recieveing file
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("week", IntegerType(), True),
    StructField("center_id", IntegerType(), True),
    StructField("city_code", IntegerType(), True),
    StructField("region_code", IntegerType(), True),
    StructField("center_type", StringType(), True),
    StructField("op_area", FloatType(), True),
    StructField("meal_id", IntegerType(), True),
    StructField("category", StringType(), True),
    StructField("cuisine", StringType(), True),
    StructField("checkout_price", FloatType(), True),
    StructField("base_price", FloatType(), True),
    StructField("emailer_for_promotion", IntegerType(), True),
    StructField("homepage_featured", IntegerType(), True),
    StructField("num_orders", FloatType(), True)
])

#reading the data stream
df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "nigd") \
        .option("failOnDataLoss", "false") \
        .load() \
        .select(from_json(col("value").cast("string"), schema).alias("data"))


schema = df.select("data.*").schema
print('&&&&&&&&&&&&&&&& ReadStream: ', schema, ' &&&&&&&&&&&Columns', df.columns)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', df ,'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
flat_df = df.select("data.id", "data.week", "data.center_id", "data.city_code", "data.region_code",
                    "data.center_type", "data.op_area", "data.meal_id", "data.category", "data.cuisine",
                    "data.checkout_price", "data.base_price", "data.emailer_for_promotion",
                    "data.homepage_featured", "data.num_orders")
query = flat_df.writeStream \
    .outputMode("append") \
    .format("csv") \
    .trigger(processingTime='10 seconds') \
    .option("path", raw_data_path ) \
    .option("checkpointLocation", checkpoint_path) \
    .option("header", "false") \
    .start()
merge()
query.stop()





