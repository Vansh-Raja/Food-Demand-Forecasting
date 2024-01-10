from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, IndexToString
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, MinMaxScaler, RobustScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import col,when
from pyspark.sql.functions import col, lit, coalesce
import pandas as pd



def preprocess(df):
    
    spark=SparkSession.builder.appName('Dataframe').getOrCreate()

    df_pyspark=spark.read.csv('train1.csv',header=True,inferSchema=True)
    
    df_pyspark = df_pyspark.drop('id')

    # categorical columns
    categorical_columns = [col[0] for col in df_pyspark.dtypes if col[1] == 'string']

    
    # Create a list of StringIndexer stages for each categorical column
    indexer_stages = [StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index") for col_name in categorical_columns]

    # Create a pipeline with all the StringIndexer stages
    pipeline = Pipeline(stages=indexer_stages)

    # Fit the pipeline on the data
    indexer_model = pipeline.fit(df_pyspark)

    # Transform the data using the fitted pipeline
    df_pyspark = indexer_model.transform(df_pyspark)
    # drop the categorcial columns
    df_pyspark = df_pyspark.drop(*categorical_columns)

    # Assuming your DataFrame is named 'df'
    df_pyspark = df_pyspark.select(
        "week",
        "city_code",
        "region_code",
        "op_area",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
        "center_type_index",
        "category_index",
        "cuisine_index",
        "num_orders"  # Move 'num_orders' to the last position
    )

    # Standardize all the columns
    feature_columns = df_pyspark.columns


    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_pyspark = assembler.transform(df_pyspark)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_pyspark)
    df_pyspark = scaler_model.transform(df_pyspark)

    #df_target = df_pyspark.select("scaled_features")

    #df_pyspark.select("scaled_features").show(truncate=False)

    df_pyspark = df_pyspark.select("scaled_features")

    
        # conversion to pandas

    pandas_df = df_pyspark.select("scaled_features").toPandas()

    # Extract individual columns from the array column 'scaled_features'
    scaled_features_columns = ["feature_" + str(i) for i in range(len(pandas_df["scaled_features"][0]))]

    # Create individual columns in the Pandas DataFrame
    pandas_df[scaled_features_columns] = pd.DataFrame(pandas_df["scaled_features"].tolist(), index=pandas_df.index)

    # Drop the original array column 'scaled_features'
    pandas_df = pandas_df.drop(columns=["scaled_features"])
    pandas_df.columns = [
        "week", "city_code", "region_code", "op_area", "checkout_price", "base_price",
        "emailer_for_promotion", "homepage_featured", "center_type_index", "category_index",
        "cuisine_index", "num_orders"
    ]

    return pandas_df


df = preprocess('train1.csv')
print(df)
