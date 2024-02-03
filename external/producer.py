import pandas as pd
from confluent_kafka import Producer
import logging
import os
import schedule
import time
import json

#reading the csv file
df=pd.read_csv(f'D:\\uni\\Python\\Pranav_Gryffindor_script\\ETL1 (2)\\ETL1\\external\\test_data.csv')

last_row_file = 'last_row.txt'
num_batches = int(input("Enter the number of batches to send: "))

batch_size = 10000

# Read the last processed row from the text file1
try:
    with open(last_row_file, 'r') as f:
        last_processed_row = int(f.read())
except:
    last_processed_row = 0

result_df = pd.DataFrame()


for i in range(num_batches):
    chunk = next(pd.read_csv('D:\\uni\\Python\\Pranav_Gryffindor_script\\ETL1 (2)\\ETL1\\external\\test_data.csv', skiprows=range(1, last_processed_row + 1), chunksize=batch_size), None)
    if chunk is not None:
        # Append the chunk to the result DataFrame
        result_df = pd.concat([result_df, chunk], ignore_index=True)
        last_processed_row += len(chunk)
        with open(last_row_file, 'w') as f:
            f.write(str(last_processed_row))
topic = 'nigd'
bootstrap_servers = 'localhost:9092'

conf = {'bootstrap.servers': bootstrap_servers,'acks': 'all',  
    'compression.type': 'gzip',
       "queue.buffering.max.messages": 10000000,}
producer = Producer(conf)

def task():
    for _, row in df.iterrows():
        message = json.dumps(row.to_dict())  # Convert the row to a dictionary and then to JSON
        producer.produce(topic, value=message)

    producer.flush()

schedule.every(5).seconds.do(task)

while True:
    schedule.run_pending()
    time.sleep(1)
task()
    