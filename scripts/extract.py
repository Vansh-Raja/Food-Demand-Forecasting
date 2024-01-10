from confluent_kafka import Consumer
import pandas as pd

raw_data_path = 'D:/Study/INTERNSHIP/ETL1/data/raw/train1.csv'

topic = 'bigd'
bootstrap_servers = 'localhost:9092'
# Kafka consumer configuration
conf = {'bootstrap.servers': bootstrap_servers, 'group.id': 'my_consumer_group'}
consumer = Consumer(conf)
consumer.subscribe([topic])
rows = []
try:
    while True:
        messages = consumer.consume(1000000, timeout=0.6)  # Consume a batch of up to 100 messages with a timeout of 0.1 seconds
        if not messages:
            continue

        for message in messages:
            if message.error():
                if message.error().code():
                    continue
                else:
                    print("Consumer error: {}".format(message.error()))
                    break


            message_value = message.value().decode('utf-8')
            row_data = message_value.split(',')

            # Append the row to the list
            rows.append(row_data)

        df = pd.DataFrame(rows, columns=['id', 'week', 'center_id', 'city_code', 'region_code', 'center_type', 'op_area', 'meal_id', 'category','cuisine','checkout_price','base_price','emailer_for_promotion','homepage_featured','num_orders'])
        break
    
except KeyboardInterrupt:
    pass
finally:
    consumer.close()

df.to_csv(raw_data_path, index=False)