from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, col, desc
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType

# Create the Spark Session
spark = SparkSession.builder \
    .appName("Read MongoDB Data") \
    .config("spark.streaming.stopGracefullyOnShutdown", True) \
    .master("local[*]") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()

# Mongo COnnection
mongo_uri = "mongodb+srv://ayush:ayush123@cluster0.eeclfqm.mongodb.net"
database_name = "PrepMong"
collection_name = "finaldata"  

#SChema
dataSchema = StructType([
    StructField("id", StringType()),
    StructField("week", StringType()),
    StructField("center_id", StringType()),
    StructField("city_code", StringType()),
    StructField("region_code", StringType()),
    StructField("center_type", StringType()),
    StructField("op_area", StringType()),
    StructField("meal_id", StringType()),
    StructField("category", StringType()),
    StructField("cuisine", StringType()),
    StructField("checkout_price", StringType()),
    StructField("base_price", StringType()),
    StructField("emailer_for_promotion", StringType()),
    StructField("homepage_featured", StringType()),
    StructField("num_orders", StringType())
])

df_mongo = spark.read.format("com.mongodb.spark.sql.DefaultSource") \
    .option("uri", mongo_uri) \
    .option("database", database_name) \
    .option("collection", collection_name) \
    .schema(dataSchema) \
    .load()


df_mongo.groupBy('week').count().sort(desc('count'))


csv_path = "final_data_dvc/food.csv"


df_mongo.write.csv(csv_path, header=True, mode="overwrite")


spark.stop()