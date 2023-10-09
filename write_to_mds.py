# Databricks notebook source
# MAGIC %md ### Writing a Spark DataFrame to MDS format (Mosaic Data Shard)
# MAGIC
# MAGIC This notebook was developed on ML DBR 14.0. It downloads the training and testing [mosaicml/instruct-v3](https://huggingface.co/datasets/mosaicml/instruct-v3) datasets from the huggingface datasets hub and converts them to Spark Dataframes. Then, MosaicML's streaming library is used to [convert the dataframes](https://docs.mosaicml.com/projects/streaming/en/stable/examples/spark_dataframe_to_MDS.html) to [MDS format](https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/dataset_format.html#formats) persisted in cloud storage.  
# MAGIC
# MAGIC The cloud storage path can then be referenced in a MosaicML fine tuning yaml config file. The MDS files will be temporarily copied into the MosaicML compute plane during model training. Trained model artifacts will then be persisted back to a cloud storage path defined in the yaml config.
# MAGIC
# MAGIC NOTE: This example currently persists MDS datasets to DBFS but they will need to be persisted to an S3 bucket that the MosaicML control plane can access using [AWS Access Keys](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/s3.html#aws-s3).

# COMMAND ----------

# MAGIC %pip install git+https://github.com/mosaicml/streaming.git@703afa6cda5d53fcefca45b3f0534918ea8e6247

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from collections import namedtuple
from datasets import load_dataset
from pyspark.sql.types import StructType, StructField, StringType
from streaming.base.converters import dataframeToMDS

# COMMAND ----------

def datasets_to_dataframes():
  """
  Given a transformers datasets name, download the dataset and 
  return Spark DataFrame versions. Result include train and test
  dataframes as well as a dataframe of label index to string
  representation.
  """

  spark_datasets = namedtuple("spark_datasets", "train test")
  
  # Define Spark schemas
  schema = StructType([StructField("prompt", StringType(), False),
                       StructField("response", StringType(), False),
                       StructField("source", StringType(), False)])

  dataset_name = 'mosaicml/instruct-v3'
  dataset = load_dataset(dataset_name)
  
  train_pd  = dataset['train'].to_pandas()
  test_pd  =  dataset['test'].to_pandas()

  def to_spark_df(pandas_df):
    """
    Convert a Pandas DataFrame to a SparkDataFrame and convert the date
    columns from a string format to a date format.
    """
    spark_df = (spark.createDataFrame(pandas_df, schema=schema))
    return spark_df
  
  train = to_spark_df(train_pd)
  test = to_spark_df(test_pd)

  return spark_datasets(train, test)

# COMMAND ----------

train, test = datasets_to_dataframes()

# COMMAND ----------

display(test)

# COMMAND ----------

# MAGIC %md #### Persist to MDS

# COMMAND ----------

# Configure MDS writer
def to_db_path(path):
  return path.replace('/dbfs/', 'dbfs:/')

path = '/dbfs/Users/marshall.carter@databricks.com/mosaicml/streaming/instruct-v3'
train_path = f'{path}/train'
test_path = f'{path}/test'
path_db = to_db_path(path)
train_path_db = to_db_path(train_path)
test_path_db = to_db_path(test_path)

columns = {'prompt': 'str', 'response': 'str'}

train_kwargs = {'out': train_path, 
                'columns': columns}
          
test_kwargs = train_kwargs.copy()
test_kwargs['out'] = test_path

# Remove exist MDS files if they exist
try:
  if dbutils.fs.ls(train_path_db) or dbutils.fs.ls(test_path_db):
    dbutils.fs.rm(path_db, recurse=True)
except:
  dbutils.fs.mkdirs(train_path)

def write_to_mds(df, kwargs):
  dataframeToMDS(df.repartition(8), 
                merge_index=True,
                mds_kwargs=kwargs)

write_to_mds(train, train_kwargs)
write_to_mds(test, test_kwargs)

# COMMAND ----------

dbutils.fs.ls(train_path_db)
