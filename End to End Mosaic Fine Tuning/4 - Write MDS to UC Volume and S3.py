# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC #  Write Gold Table to MDS (Mosaic Data Shard) and JSONL formats to UC Volume and/or S3
# MAGIC
# MAGIC -----
# MAGIC #### Overview
# MAGIC
# MAGIC Reads standard Gold table ready for training and converts the data to MDS and JSONL format for model fine tuning. Mosaic and Databricks supports writing the datasets via 2 options: 
# MAGIC
# MAGIC <b> Option 1: UC Volume </b> Write directly to a governed Volume in Unity Catalog
# MAGIC
# MAGIC <b> Option 2: External S3 </b> Writer to any S3 external bucket
# MAGIC
# MAGIC -----
# MAGIC Use of the MDS and JSONL versions of the datasets depends on which MosaicML training API is used. For the standard API (mcli run), the MDS format should be used. For the finetune API, the JSONL version of the datasets must be used; this is because the finetune API handles the conversion to MDS itself. There is currently no option to pass MDS datasets directly to the finetuning API.  
# MAGIC
# MAGIC The finetuning API is a higher-level API and automatically saves models in a huggingface transformers compatible format, making it easy to load the tuned model in Databricks and log it to MLFlow.

# COMMAND ----------

# MAGIC %pip install 'mosaicml-streaming[databricks]>=0.6,<0.7'

# COMMAND ----------

#%pip install git+https://github.com/mosaicml/streaming.git@703afa6cda5d53fcefca45b3f0534918ea8e6247 s3fs

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from collections import namedtuple
from datasets import load_dataset
from pyspark.sql.types import StructType, StructField, StringTypec
from streaming.base.converters import dataframeToMDS

# COMMAND ----------

# DBTITLE 1,Load Clean Final Data Set
all_dataframe = spark.table("main.mosaic_end_to_end.gold_training_final_set").select("prompt", "response")

train, test = all_dataframe.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

display(all_dataframe)

# COMMAND ----------

# MAGIC %md # Persist to MDS
# MAGIC
# MAGIC ### 2 Options
# MAGIC
# MAGIC 1. Write to UC Managed Volumes to have all source data governed and managed in one place
# MAGIC 2. Write to External S3 Bucket anywhere

# COMMAND ----------

# DBTITLE 1,Convert Spark Data Frames to MDS
# Configure MDS writer
def to_db_path(path):
  return path.replace('/dbfs/', 'dbfs:/')

local_path = 'dbfs:/Users/cody.davis@databricks.com/mosaicml/fine_tuning'

path = 's3://codymosaic/datasets'
train_path = f'{path}/train'
test_path = f'{path}/test'

columns = {'prompt': 'str', 'response': 'str'}

train_kwargs = {'out': train_path, 
                'columns': columns}
          
test_kwargs = train_kwargs.copy()
test_kwargs['out'] = test_path

# Remove exist MDS files if they exist
try:
  if dbutils.fs.ls(train_path) or dbutils.fs.ls(test_path):
    dbutils.fs.rm(path, recurse=True)

except:
  dbutils.fs.mkdirs(train_path)
  dbutils.fs.mkdirs(test_path)

def write_to_mds(df, kwargs):
  dataframeToMDS(df.repartition(8), 
                merge_index=True,
                mds_kwargs=kwargs

  )


# COMMAND ----------

print(train_path)
print(test_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Option 1: Persisting MDS to UC Volumes

# COMMAND ----------

# DBTITLE 1,Write to Managed or External UC Volume
# UC Volume path
train_volume_path = "/Volumes/main/mosaic_end_to_end/mosaic_source_data/train"


columns = {'prompt': 'str', 'response': 'str'}

train_kwargs = {'out': train_volume_path, 
                'columns': columns}
          

## Write Training Data Frame to Volume
dataframeToMDS(train.repartition(8), 
                merge_index=True,
                mds_kwargs=train_kwargs

  )

# COMMAND ----------

# DBTITLE 1,Write Test Data Set to UC Managed Volume
# UC Volume path
test_volume_path = "/Volumes/main/mosaic_end_to_end/mosaic_source_data/test"


columns = {'prompt': 'str', 'response': 'str'}

test_kwargs = {'out': test_volume_path, 
                'columns': columns}
          

## Write Training Data Frame to Volume
dataframeToMDS(test.repartition(8), 
                merge_index=True,
                mds_kwargs=test_kwargs

  )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Option 2: Write to External S3 Bucket

# COMMAND ----------

# DBTITLE 1,Write Training Set to External Bucket
write_to_mds(train, train_kwargs)

# COMMAND ----------

# DBTITLE 1,Write Test Set to External Bucket
write_to_mds(test, test_kwargs)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Write to JSONL to External Bucket

# COMMAND ----------

train_pd = train.select("prompt", "response").toPandas()
test_pd = test.select("prompt", "response").toPandas()

# COMMAND ----------

train_pd.to_json("s3://codymosaic/datasets/jsonl/train.jsonl",
                 orient="records",
                 lines=True)

test_pd.to_json("s3://codymosaic/datasets/jsonl/test.jsonl",
                 orient="records",
                 lines=True)

# COMMAND ----------

dbutils.fs.ls("s3://codymosaic/datasets/")