
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType

output_path = '/home/gowtham/Documents/Python/BigData/spark_demo/output'
checkpoint_path = '/home/gowtham/Documents/Python/BigData/spark_demo/checkpoint'

# Spark session
spark = SparkSession.builder \
    .appName("Kafka_consumer_demo") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0") \
    .getOrCreate()

# Kafka broker address
kafka_bootstrap_servers = 'localhost:9092'
# Kafka topic
kafka_topic = 'spark-streaming-2'

# Define the schema for the incoming data
schema = StructType([StructField("timestamp", StringType(), True),
                     StructField("iss_position", StructType([
                         StructField("latitude", StringType(), True),
                         StructField("longitude", StringType(), True)
                     ]), True)])

# Create DataFrame to read from Kafka topic
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .load()


# Parse JSON data and select relevant columns
parsed_df = df.selectExpr("CAST(value AS STRING)").select(from_json("value", schema).alias("data")).select("data.*")

# Flatten the 'iss_position' struct column
flattened_df = parsed_df.select("timestamp", "iss_position.latitude", "iss_position.longitude")

# Write the streaming data to a CSV sink (append mode)
query = flattened_df.writeStream \
    .outputMode("append") \
    .format("csv") \
    .trigger(processingTime='20 seconds') \
    .option("path", output_path) \
    .option("checkpointLocation", checkpoint_path) \
    .option("header", "true") \
    .start()

# Wait for the streaming query to finish
query.awaitTermination()

# Stop the Spark session
spark.stop()
