import json
import requests
from kafka import KafkaProducer
from time import sleep

producer=KafkaProducer(bootstrap_servers=['localhost:9092'],
                      value_serializer=lambda x: json.dumps(x).encode('utf-8')
                      )

kafka_topic = 'spark-streaming-2'

for i in range(50):
    response=requests.get('http://api.open-notify.org/iss-now.json')
    #data=json.loads(res.content.decode('utf-8'))
    data = response.json()
    print('Data published to topics: ', data)
    producer.send(kafka_topic, value=data)       # Topic name
    sleep(5)
    #producer.flush()


