# Databricks notebook source
# DBTITLE 1,a
# MAGIC %md
# MAGIC # Optimized MPT serving example
# MAGIC
# MAGIC Optimized LLM Serving enables you to take your fine-tuned LLM from MosaicML and deploy them on Databricks Model Serving with automatic optimizations for improved latency and throughput on GPUs. Currently, Databricks supports optimizations for Llama2 and Mosaic MPT class of models.
# MAGIC
# MAGIC This example walks through:
# MAGIC
# MAGIC 1. Loading the fine-tuned model in Hugging Face `transformers` format from DBFS
# MAGIC 2. Logging the model in an optimized serving supported format into the Databricks Unity Catalog or Workspace Registry
# MAGIC 3. Enabling optimized serving on the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC - Attach a cluster with sufficient memory to the notebook
# MAGIC - Make sure to have MLflow version 2.7.0 or later installed
# MAGIC - Make sure to enable **Models in UC**, especially when working with models larger than 7B in size
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 1: Log the model for optimized LLM serving

# COMMAND ----------

# Update and install required dependencies
!pip install -U mlflow
!pip install -U transformers
!pip install -U accelerate
dbutils.library.restartPython()

# COMMAND ----------

############ Your input starts here ############

# Provide the url path to your remote model checkpoint directory
# - example s3 model path: "s3://bucket_name/model_name/hf_checkpoints"
# - example gcp model path: "gs://bucket_name/model_name/hf_checkpoints" TBD
# - example dbfs model path: "/dbfs/home/model_name/hf_checkpoints"
MODEL_CHECKPOINT_PATH = '/dbfs/Users/marshall.carter@databricks.com/mosaicml/models_v2_hf'

# Provide a model name you want to use to register the model in the model registry
# - example: "mpt_7b"
# Attention: If using unity catalog to register model, the model name should follow thi format: `catalog_name.schema_name.model_name`
# - example: "main.tianshu.test_mpt_7b"
CATALOG = "main"
SCHEMA = "mosaic_end_to_end"
MODEL_NAME = "mpt-7b-instruct-custom"

# Set model precision for your new model endpoint, choose from 'float32',  'float16', 'bfloat16', we recommend 'bfloat16'
MODEL_PRECISION = 'bfloat16'


# Set to true if you have enabled unity catalog in your workspace and want to use unity catalog to register your model
# USE_UNITY_CATALOG = False
USE_UNITY_CATALOG = True

############  Your input ends here  ############

if USE_UNITY_CATALOG:
  REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
else:
  REGISTERED_MODEL_NAME = MODEL_NAME

############  DO NOT MODIFY  ############

LOCAL_CHECKPOINT_PATH = "/tmp/mosaicml/"
LOCAL_MODEL_NAME = "local_model"

############  DO NOT MODIFY  ############

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
import numpy as np
import os
import torch


MODEL_REGISTRATION_TIMEOUT = 1200

print(f"⏳ Loading your model checkpoint...")

MODEL_PRECISION_MAP = {
  'float32': torch.float32,
  'float16': torch.float16,
  'bfloat16': torch.bfloat16,
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT_PATH, trust_remote_code=True, torch_dtype=MODEL_PRECISION_MAP[MODEL_PRECISION])

print(f"✅ Done!")

# COMMAND ----------

# MAGIC %md
# MAGIC To enable optimized serving, when logging the model, include the extra metadata dictionary when calling `mlflow.transformers.log_model` as shown below:
# MAGIC
# MAGIC ```
# MAGIC metadata = {"task": "llm/v1/completions"}
# MAGIC ```
# MAGIC This specifies the API signature used for the model serving endpoint.
# MAGIC

# COMMAND ----------

# Log the model with its details such as artifacts, pip requirements and input example

# Define the model input and output schema
input_schema = Schema([
    ColSpec("string", "prompt"),
    ColSpec("double", "temperature", optional=True),
    ColSpec("integer", "max_tokens", optional=True),
    ColSpec("string", "stop", optional=True),
    ColSpec("integer", "candidate_count", optional=True)
])

output_schema = Schema([
    ColSpec('string', 'predictions')
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define an example input
input_example = {
    "prompt": np.array([
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "What is Apache Spark?\n\n"
        "### Response:\n"
    ]),
    "max_tokens": np.array([75]),
    "temperature": np.array([0.0])
}

# Start a new MLflow run
print(f"⏳ Registering your model registry...")

if USE_UNITY_CATALOG:
  # Log the model to the unity catalog, since that is much faster.
  mlflow.set_registry_uri("databricks-uc")
  with mlflow.start_run() as run:
      components = {
          "model": model,
          "tokenizer": tokenizer,
      }
      mlflow.transformers.log_model(
          transformers_model=components,
          artifact_path="model",
          registered_model_name=REGISTERED_MODEL_NAME,
          signature=signature,
          input_example=input_example,
          metadata={"task": "llm/v1/completions"},
          task='text-generation'
      )
else:
  with mlflow.start_run() as run:
      components = {
          "model": model,
          "tokenizer": tokenizer,
      }
      mlflow.transformers.log_model(
          transformers_model=components,
          artifact_path="model",
          registered_model_name=REGISTERED_MODEL_NAME,
          signature=signature,
          input_example=input_example,
          metadata={"task": "llm/v1/completions"},
          task='text-generation',
          await_registration_for=MODEL_REGISTRATION_TIMEOUT
      )

print(f"✅ Done!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configure and create your model serving GPU endpoint
# MAGIC
# MAGIC Modify the cell below to change the endpoint name. After calling the create endpoint API, the logged MPT-7B model is automatically deployed with optimized LLM serving.

# COMMAND ----------

# MAGIC %md ### Set latest model version to production (UC-only)
# MAGIC
# MAGIC Otherwise, get latest model version

# COMMAND ----------

client = mlflow.MlflowClient()

if USE_UNITY_CATALOG:
  mlflow.set_registry_uri("databricks-uc")
else:
  mlflow.set_registry_uri("databricks")

def get_latest_version_number(model_name: str) -> int:
    versions = client.search_model_versions(f"name='{model_name}'")
    version_numbers = [int(v.version) for v in versions]
    latest_version = max(version_numbers)
    return latest_version

latest_version = get_latest_version_number(REGISTERED_MODEL_NAME)
print(f"latest model version: {latest_version}")

if USE_UNITY_CATALOG:
  client.set_registered_model_alias(REGISTERED_MODEL_NAME, "Champion", latest_version)

# COMMAND ----------

# Set the name of the MLflow endpoint
endpoint_name = 'mpt-7b-instruct-e2e-custom'

# Name of the registered MLflow model
model_name = REGISTERED_MODEL_NAME 

# Get the latest version of the MLflow model
# model_version = 1

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
# workload_type = "GPU_MEDIUM_4" 
workload_type = "GPU_MEDIUM" 

# Specify the compute scale-out size(Small, Medium, Large, etc.)
workload_size = "Small" 

# Specify Scale to Zero(only supported for CPU endpoints)
scale_to_zero = False 

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

import requests
import json

def get_endpoint_status(endpoint_name:str) -> dict:
    response = requests.get(url=f"{API_ROOT}/api/2.0/serving-endpoints/{endpoint_name}", json=data, headers=headers)
    # print(json.dumps(response.json(), indent=4))
    return response.json()

def create_endpoint(data:dict, headers:dict) -> dict:
    response = requests.post(url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers)
    # print(json.dumps(response.json(), indent=4))
    return response.json()

def update_endpoint(endpoint_name:str, data:dict, headers:dict) -> dict:
    response = requests.put(url=f"{API_ROOT}/api/2.0/serving-endpoints/{endpoint_name}/config", json=data["config"], headers=headers)
    # print(json.dumps(response.json(), indent=4))
    return response.json()

def create_or_update_endpoint(endpoint_name:str, data:dict, headers:dict) -> dict:
    status_check = get_endpoint_status(endpoint_name)
    if "state" in status_check:
        print(f"updating endpoint {endpoint_name} to latest version")
        response = update_endpoint(endpoint_name, data, headers)
    else:
        print(f"creating new endpoint")
        response = create_endpoint()
    print(json.dumps(response, indent=4))
    return response

# COMMAND ----------

data = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": latest_version,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
                "workload_type": workload_type,
            }
        ]
    },
}

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

status = create_or_update_endpoint(endpoint_name=endpoint_name, data=data, headers=headers)
print(status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View your endpoint
# MAGIC To see your more information about your endpoint, go to the **Serving** UI section on the left navigation bar and search for your endpoint name.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Query your endpoint
# MAGIC
# MAGIC Once your endpoint is ready, you can query it by making an API request. Depending on the model size and complexity, it can take 30 minutes or more for the endpoint to get ready.  

# COMMAND ----------

import requests
import json

def query_endpoint(prompt:str) -> dict:
    data = {
        "inputs": {
            "prompt": [prompt]
        },
        "params": {
            "max_tokens": 200, 
            "temperature": 0.0
        }
    }
    headers = {
        "Context-Type": "text/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    response = requests.post(
        url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations",
        json=data,
        headers=headers
    )

    # print(json.dumps(response.json()))

    return response.json()

# COMMAND ----------

prompt1 = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is Apache Spark?\n\n### Response:\n"

results1 = query_endpoint(prompt1)
print(results1["predictions"][0]["candidates"][0]["text"])

# COMMAND ----------

prompt2 = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a recursive Python function.\n\n### Response:\n"

results2 = query_endpoint(prompt2)
print(results2["predictions"][0]["candidates"][0]["text"])

# COMMAND ----------

prompt3 = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a Python function that finds the most rare record in a dataset.\n\n### Response:\n"

results3 = query_endpoint(prompt3)
print(results3["predictions"][0]["candidates"][0]["text"])
