# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Fine tuning an LLM using MosaicML
# MAGIC
# MAGIC -----
# MAGIC #### Overview: 
# MAGIC This notebook utilizes the Mosaic fine-tuning API to kick off a trigger a LLM tuning job on the Databricks/Mosaic Platform. The fine tuning API can reference governed UC Volumes or read training/test data directly from S3. 

# COMMAND ----------

# DBTITLE 1,Install Mosaic CLI
# MAGIC %pip install -q mosaicml-cli

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mcli
from mcli import finetune

# COMMAND ----------

# DBTITLE 1,Get Mosaic API Key in Databricks Secret
mcli.set_api_key(
  dbutils.secrets.get(scope="mosaicml", key="api_key")
                 )

# COMMAND ----------

# DBTITLE 1,Kick off Fine Tuning run on Mosaic Platform Read from S3 or UC Volume Directly
## Insert UC Volume or your own S3 Paths here

run = finetune(
    model="mosaicml/mpt-7b",
    train_data_path="s3://mosaic-streaming/datasets/train.jsonl",
    save_folder="s3://mosaic-streaming/fine_tune_api/",
    eval_data_path="s3://mosaic-streaming/datasets/test.jsonl",
    training_duration="3ep"
)

# COMMAND ----------

# DBTITLE 1,Show Run Details
run

# COMMAND ----------

# DBTITLE 1,Show Run Status
mcli.get_run(run.name).status

# COMMAND ----------

# DBTITLE 1,Track and Print Run Logs
logs = mcli.follow_run_logs(run.name)

for line in logs:
  print(line)