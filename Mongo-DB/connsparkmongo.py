from pyspark.sql import SparkSession

# Create a PySpark session
spark = SparkSession.builder.appName("MongoDBExample").getOrCreate()

# MongoDB connection options
mongo_uri = "mongodb://localhost:27017"
mongo_database = "PrepodSparkstreaming"
mongo_collection = "train1"

df = spark.read.format("mongo") \
    .option("uri", mongo_uri) \
    .option("database", mongo_database) \
    .option("collection", mongo_collection) \
    .load()

df.show()

df.write.format("mongo") \
    .mode("append") \
    .option("uri", mongo_uri) \
    .option("database", mongo_database) \
    .option("collection", mongo_collection) \
    .save()
