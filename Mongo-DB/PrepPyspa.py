from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Create a Spark session
spark = SparkSession.builder \
    .appName("Prepo") \
    .config("spark.sql.streaming.checkpointLocation", "C:\\Users\\ayush\\pythonProject2023\\Prepod\\checkpoint") \
    .config("spark.local.dir", "C:\\Users\\ayush\pythonProject2023\\Prepod\\directory") \
    .getOrCreate()


schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("week", IntegerType(), True),
    StructField("center_id", IntegerType(), True),
    StructField("meal_id", IntegerType(), True),
    StructField("checkout_price", DoubleType(), True),
    StructField("base_price", DoubleType(), True),
    StructField("emailer_for_promotion", IntegerType(), True),
    StructField("homepage_featured", IntegerType(), True),
    StructField("num_orders", IntegerType(), True)
])

csv_file_path = "train1.csv"

csv_stream_df = spark.readStream \
    .option("header", "true") \
    .schema(schema) \
    .csv(csv_file_path)

output_csv_path = "output_csv"

write_query = csv_stream_df.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path", output_csv_path) \
    .option("checkpointLocation", "C:\\Users\\ayush\\pythonProject2023\\Prepod\\checkpoint") \
    .start()

write_query.awaitTermination()
