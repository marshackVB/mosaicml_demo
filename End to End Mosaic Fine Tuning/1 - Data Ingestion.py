# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Mosaic End to End Data Pipeline for Fine Tuning Demo
# MAGIC
# MAGIC -----
# MAGIC ## Data Ingestion
# MAGIC
# MAGIC #### Overview:
# MAGIC This series shows the end-to-end process of ingesting a custom data set to fine tune a foundation model. The demo goes through data ingestion, cleaning, deduping, model fine-tuning, model registration, and model serving all in Databricks on Unity Catalog. 
# MAGIC
# MAGIC <b> Steps: </b>
# MAGIC
# MAGIC 1. Ingest 2 Datasets from Hugging Face
# MAGIC 2. Creaate Database Schema
# MAGIC 3. Load Bronze Tables of each dataset (or stream)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install 'mosaicml-streaming[databricks]>=0.6,<0.7'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Create Database for Pipeline
# MAGIC %sql
# MAGIC
# MAGIC CREATE DATABASE IF NOT EXISTS main.mosaic_end_to_end;
# MAGIC USE CATALOG main;
# MAGIC
# MAGIC USE SCHEMA mosaic_end_to_end;

# COMMAND ----------

from collections import namedtuple
from datasets import load_dataset
from streaming.base.converters import dataframeToMDS
from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Ingest Data 
# MAGIC
# MAGIC 1. from streaming source
# MAGIC 2. Apps
# MAGIC 3. Kafka
# MAGIC 4. Kinesis
# MAGIC 5. JDBC
# MAGIC 6. REST API

# COMMAND ----------

# DBTITLE 1,Data Set for Python Code Gen
dataset_code_gen_name = "iamtarun/python_code_instructions_18k_alpaca"

dataset_code_gen = load_dataset(dataset_code_gen_name)

# COMMAND ----------

# DBTITLE 1,Mosaic Instruct Dataset
dataset_name = 'mosaicml/instruct-v3'
dataset = load_dataset(dataset_name)

# COMMAND ----------

dataset_code_gen

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Pre Processing
# MAGIC

# COMMAND ----------

# DBTITLE 1,Convert to Spark Data Frames
def datasets_to_dataframes(dataset):
  """
  Given a transformers datasets name, download the dataset and 
  return Spark DataFrame versions. Result include train and test
  dataframes as well as a dataframe of label index to string
  representation.
  """

  spark_datasets = namedtuple("spark_datasets", "train test")
  
  # Define Spark schemas
  new_schema = StructType()
  extracted_cols = dataset["train"].features

  for i in extracted_cols:
    new_schema.add(i, data_type="string")


  train_pd  = dataset['train'].to_pandas()

  try:
    test_pd  =  dataset['test'].to_pandas()

  except: 
    test_pd = None

  def to_spark_df(pandas_df):
    """
    Convert a Pandas DataFrame to a SparkDataFrame and convert the date
    columns from a string format to a date format.
    """
    spark_df = (spark.createDataFrame(pandas_df, schema=new_schema))
    return spark_df
  
  train = to_spark_df(train_pd)
  if test_pd is not None:
    test = to_spark_df(test_pd)
    return spark_datasets(train, test)
  
  else:
    return train

# COMMAND ----------

spark_dataframes_code_gen = datasets_to_dataframes(dataset_code_gen)
spark_dataframes_code_gen = (spark.createDataFrame(spark_dataframes_code_gen.rdd)
   .withColumn("update_timestamp", current_timestamp())                              
)

# COMMAND ----------

display(spark_dataframes_code_gen)

# COMMAND ----------

spark_dataframe_instruct_train,  spark_dataframe_instruct_test = datasets_to_dataframes(dataset)

spark_dataframe_instruct_train = (spark.createDataFrame(spark_dataframe_instruct_train.rdd)
 .withColumn("update_timestamp", current_timestamp())                                 
)
spark_dataframe_instruct_test = (spark.createDataFrame(spark_dataframe_instruct_test.rdd)
  .withColumn("update_timestamp", current_timestamp())                                     
)

# COMMAND ----------

display(spark_dataframe_instruct_train)

# COMMAND ----------

# DBTITLE 1,Create Target Bronze Tables to Load or Stream Into
# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE TABLE bronze_mosaic_instruct_data_feed
# MAGIC (
# MAGIC   prompt STRING,
# MAGIC   response STRING,
# MAGIC   source STRING,
# MAGIC   update_timestamp TIMESTAMP
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

# DBTITLE 1,Write or Stream to Bronze Delta Tables
spark_dataframe_instruct_train.write.format("delta").mode("append").saveAsTable("bronze_mosaic_instruct_data_feed")
spark_dataframe_instruct_test.write.format("delta").mode("append").saveAsTable("bronze_mosaic_instruct_data_feed")
spark_dataframes_code_gen.write.format("delta").mode("append").saveAsTable("bronze_python_code_gen_data_feed")