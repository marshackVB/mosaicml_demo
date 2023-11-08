# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Data Clean and Prep to Standard Format
# MAGIC -----
# MAGIC
# MAGIC #### Overview and Steps
# MAGIC 1. Consolidate multiple data sources into single "training" and "testing" dataset
# MAGIC 2. Clean PII by adding other rule-based or model-based filters in this pipeline
# MAGIC 3. Remove "bad" characters
# MAGIC 4. Standardize Prompt and Response formatting across data sets with Registered UC Function
# MAGIC 4. Save to clean / silver data set for model training / fine tuning 

# COMMAND ----------

# MAGIC %pip install 'mosaicml-streaming[databricks]>=0.6,<0.7'
# MAGIC %pip install torch

# COMMAND ----------

# DBTITLE 1,Install functions to clean PII with NER
# MAGIC %pip install transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


from collections import namedtuple
from datasets import load_dataset
from pyspark.sql.types import StructType, StructField, StringType
from streaming.base.converters import dataframeToMDS

# COMMAND ----------

# DBTITLE 1,Scope Notebook to use Database
# MAGIC %sql
# MAGIC
# MAGIC CREATE DATABASE IF NOT EXISTS main.mosaic_end_to_end;
# MAGIC USE CATALOG main;
# MAGIC
# MAGIC USE SCHEMA mosaic_end_to_end;

# COMMAND ----------

# DBTITLE 1,Define model for NER recognition
from transformers import pipeline


## Define NER transformer pipeline
ner = pipeline("ner", aggregation_strategy="simple")

# COMMAND ----------

# DBTITLE 1,Broadcast NER model to all workers to use in spark UDF
sc.broadcast(ner)

# COMMAND ----------

# DBTITLE 1,Define UDF to run model inference on all workers
@udf("string")
def scrub_input_string_NER(input_st):
  import re

  output = ner(input_st)
  clean_st = input_st

  ## parse output and actually clean the string

  for i, entity in enumerate(output):

    ## If pretty confident it a named person, than replace

    try:

      if entity.get("score") >= 0.8:

        clean_st = re.sub(entity.get("word"), entity.get("entity_group"), clean_st)
    except:
      ## If fail, just keep going and try to continue parsing
      pass

  return clean_st

# COMMAND ----------

# DBTITLE 1,Register to use in SQL / Spark
spark.udf.register("scrub_input_string_NER", scrub_input_string_NER)

# COMMAND ----------

# DBTITLE 1,Create Clean Table for Each Data Source
# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS clean_mosaic_instruct_data_feed
# MAGIC AS
# MAGIC SELECT * FROM bronze_mosaic_instruct_data_feed
# MAGIC WHERe 1=0;

# COMMAND ----------

# DBTITLE 1,Load Data Set to Identify Named Entities to Scrub
# MAGIC %sql
# MAGIC
# MAGIC INSERT INTO clean_mosaic_instruct_data_feed
# MAGIC SELECT 
# MAGIC scrub_input_string_NER(prompt) AS prompt,  -- scrubbing of input data for named entities
# MAGIC response,
# MAGIC source,
# MAGIC update_timestamp
# MAGIC FROM bronze_mosaic_instruct_data_feed

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Clean and Format Code-gen dataset to fit model format and standardize dataset

# COMMAND ----------

# DBTITLE 1,Create Code Gen Clean Data Source
# MAGIC %sql
# MAGIC
# MAGIC DROP TABLE IF EXISTS clean_python_code_gen_data_feed;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS clean_python_code_gen_data_feed
# MAGIC (prompt STRING,
# MAGIC response STRING,
# MAGIC source STRING,
# MAGIC update_timestamp TIMESTAMP)
# MAGIC USING DELTA
# MAGIC ;

# COMMAND ----------

# DBTITLE 1,Define Managed Custom Functions Governed in UC
# MAGIC %sql
# MAGIC
# MAGIC -- Prompt Engineering UC / SQL Function with SQL + Python functions
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION format_prompt(instruction STRING, input_context STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON 
# MAGIC AS 
# MAGIC $$
# MAGIC
# MAGIC     def format_instruction_to_prompt(instruction, input_context):
# MAGIC
# MAGIC       formatted_instruct = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction {instruction} """
# MAGIC
# MAGIC       if len(input_context) > 1:
# MAGIC         formatted_instruct += f""" Use the additional input context: {input_context} """
# MAGIC
# MAGIC       formatted_instruct += " ### Response "
# MAGIC
# MAGIC       return formatted_instruct
# MAGIC
# MAGIC     return format_instruction_to_prompt(instruction, input_context)
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## All of this can be scaled and distributed to scale

# COMMAND ----------

# DBTITLE 1,Show Cleaned Prompts
# MAGIC %sql
# MAGIC
# MAGIC SELECT format_prompt(instruction, input) AS prompt,
# MAGIC output AS response,
# MAGIC 'python_hugging_face_code_gen_data_set' AS source,
# MAGIC update_timestamp
# MAGIC FROM  bronze_python_code_gen_data_feed;

# COMMAND ----------

# DBTITLE 1,Load into Clean Table for this data source
# MAGIC %sql
# MAGIC
# MAGIC -- Clean and Format the Code Gen data to desired format for fine tuning
# MAGIC -- The fine tuning model template expects prompt, response, source, and update_timestamp
# MAGIC INSERT INTO clean_python_code_gen_data_feed
# MAGIC SELECT format_prompt(instruction, input) AS prompt,
# MAGIC output AS response,
# MAGIC 'python_hugging_face_code_gen_data_set' AS source,
# MAGIC update_timestamp
# MAGIC FROM  bronze_python_code_gen_data_feed;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Create unified standard training set for all data sources
# MAGIC %sql
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.mosaic_end_to_end.silver_all_input_data;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS main.mosaic_end_to_end.silver_all_input_data
# MAGIC (id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC prompt STRING,
# MAGIC response STRING,
# MAGIC source STRING,
# MAGIC update_timestamp TIMESTAMP)
# MAGIC USING DELTA;

# COMMAND ----------

# DBTITLE 1,Write (or Stream) all data into consolidated layer
# MAGIC %sql
# MAGIC
# MAGIC INSERT INTO main.mosaic_end_to_end.silver_all_input_data (prompt, response, source, update_timestamp)
# MAGIC SELECT * FROM main.mosaic_end_to_end.clean_mosaic_instruct_data_feed;
# MAGIC
# MAGIC
# MAGIC INSERT INTO main.mosaic_end_to_end.silver_all_input_data (prompt, response, source, update_timestamp)
# MAGIC SELECT * FROM main.mosaic_end_to_end.clean_python_code_gen_data_feed