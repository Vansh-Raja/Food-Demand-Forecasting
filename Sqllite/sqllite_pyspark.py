import sqlite3

con = sqlite3.connect('example.db')
cur = con.cursor()
# Create table
cur.execute(
    '''CREATE TABLE IF NOT EXISTS stocks
       (date text, trans text, symbol text, qty real, price real)''')
# Insert a row of data
cur.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")
# Save (commit) the changes
con.commit()

import os

from pyspark.sql import SparkSession

(SparkSession.builder
    .master("local")
    .appName("SQLite JDBC")
    .config(
        "spark.jars",
        "{}\sqlite-jdbc-3.34.0.jar".format(os.getcwd()))
    .config(
        "spark.driver.extraClassPath",
        "{}\sqlite-jdbc-3.34.0.jar".format(os.getcwd()))
    .getOrCreate())

import pyspark.pandas as ps

df = ps.read_sql("stocks", con="jdbc:sqlite:{}\example.db".format(os.getcwd()))
df

df.price += 1
df.spark.to_spark_io(
    format="jdbc", mode="append",
    dbtable="stocks", url="jdbc:sqlite:{}\example.db".format(os.getcwd()))
ps.read_sql("stocks", con="jdbc:sqlite:{}\example.db".format(os.getcwd()))