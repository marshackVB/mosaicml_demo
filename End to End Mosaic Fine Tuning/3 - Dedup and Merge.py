# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Data Clean and Prep to Standard Format
# MAGIC -----
# MAGIC
# MAGIC #### Overview and Steps
# MAGIC
# MAGIC 1. Use LSH Hashing to De-dup likely duplicate prompts, especially if reading in from multiple sources
# MAGIC 2. Quarantine Duplicates and any other data quality issues into "dups" table to investigate

# COMMAND ----------

# MAGIC %pip install ftfy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import element_at
from pyspark.ml.feature import MinHashLSH
import mlflow
import pandas as pd
import numpy as np
import hashlib
import struct
import ftfy
import string
import re

# COMMAND ----------

# DBTITLE 1,Scope Notebook to Database
# MAGIC %sql
# MAGIC
# MAGIC CREATE DATABASE IF NOT EXISTS main.mosaic_end_to_end;
# MAGIC USE CATALOG main;
# MAGIC
# MAGIC USE SCHEMA mosaic_end_to_end;

# COMMAND ----------

# DBTITLE 1,Load or Stream Source Data from Silver Layer on Delta 
silver_df = spark.table("main.mosaic_end_to_end.silver_all_input_data")

# COMMAND ----------

# DBTITLE 1,Text Standardization and Cleaning UDF - Be careful here with code gen, you might need the extra context
# > Do some NFC correction (remove non unicode characters)
# > Strip out extra whitespace
# > Some punctuation fixes
# > Lowercasing everything for dedupe

@udf('string')
def clean_and_fix_text(s):
  #s = ftfy.fix_text(s, normalization="NFC")
  s = s.lower()
  s = s.translate(str.maketrans("", "", string.punctuation))
  s = re.sub(r"\s+", " ", s.strip())
  return s

# Using 50 here for test, but should be 200 or more
threshold = 50


##standard_df = silver_df.withColumn("text", clean_and_fix_text("prompt")).filter(length("text") > threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## De-dup Dataset with Hashing Similarity Model

# COMMAND ----------

# DBTITLE 1,UDFs to perform finger-print based de-dup
## Computing a fingerprint for each text sample

def sha1_hash32(data):
  return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]

@udf(VectorUDT())
def compute_fingerprints(doc):
  n = 13
  # Throw away top 2 bits to limit size of sparse vector (will cause issues below otherwise - negative index, etc)
  fingerprints = np.unique([(sha1_hash32(doc[i-n:i].encode('utf-8')) & 0x3FFFFFFF) for i in range(n, len(doc) + 1)])
  return Vectors.sparse(0x40000000, fingerprints, np.ones(len(fingerprints)))


## UDF to convert hashes to array for optimimal de-dup join performance
@udf("array<int>")
def hashes_to_array(hash_vec_array):
  return [int(hash_vec.values[0]) for hash_vec in hash_vec_array]


# Estimate jaccard similarity from hash intersection
@udf('float')
def minhash_score(hashesA, hashesB):
  intersection = float(len(np.intersect1d(hashesA, hashesB)))
  return intersection / (len(hashesA) + len(hashesB) - intersection)



# COMMAND ----------

signature_df = silver_df.withColumn("fingerprints", compute_fingerprints("prompt"))
signature_df.createOrReplaceTempView("signature_df")

# COMMAND ----------

#display(signature_df)

# COMMAND ----------

# DBTITLE 1,Feed Hashes into MinHashLSH Model to get similarity scores
mh = MinHashLSH(inputCol="fingerprints", outputCol="hash_vecs", seed=42, numHashTables=128) # 128 matches source
model = mh.fit(signature_df)
hash_df = model.transform(signature_df)


hash_df.createOrReplaceTempView("hash_df")

# COMMAND ----------

# For simplicity, simplify the 'hashes' to just an array of ints
# Hack: pull out (arbitrarily) the first 3 hashes to be used as join keys

hash_expanded_df = (hash_df.select("prompt", "id", hashes_to_array("hash_vecs").alias("hashes")).
  withColumn("hash1", element_at("hashes", 1)).
  withColumn("hash2", element_at("hashes", 2)).
  withColumn("hash3", element_at("hashes", 3))
)

hash_expanded_df.cache()

# COMMAND ----------

#display(hash_expanded_df)

# COMMAND ----------

# DBTITLE 1,Compare Hashes to each other to look for dups based on vector fingerprints
# We will only consider pairs that match in the first 3 (of 128) hashes, which isn't a bad approximation;
# of course, similar docs may not match in any of these 3 and yet be similar!

hash_cols = [f"hash{i+1}" for i in range(3)]
join_df = hash_expanded_df.alias("a").join(hash_expanded_df.alias("b"), on=hash_cols).filter("a.id < b.id").drop(*hash_cols)

# Filter on score threshold
approx_match_df = join_df.withColumn("score", minhash_score("a.hashes", "b.hashes"))

approx_match_df.cache()
hash_expanded_df.cache()

approx_match_df.createOrReplaceTempView("hash_calculations")
hash_expanded_df.createOrReplaceTempView("hash_expanded_df")

# COMMAND ----------

# DBTITLE 1,Do Upsert of Incoming Prompts - de-duping
# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC -- First do intra-batch de-dup and quarantine
# MAGIC -- then MERGE to final table with de-duped batch on hashes
# MAGIC
# MAGIC CREATE OR REPLACE TABLE main.mosaic_end_to_end.temp_stage_hashes_dups
# MAGIC AS
# MAGIC SELECT id,
# MAGIC hash1, 
# MAGIC hash2,
# MAGIC hash3,
# MAGIC row_number() OVER (PARTITION BY hash1, hash2, hash3 ORDER BY id) AS DupRank
# MAGIC FROM hash_expanded_df;
# MAGIC
# MAGIC
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.mosaic_end_to_end.silver_de_duped_hashed;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS main.mosaic_end_to_end.silver_de_duped_hashed
# MAGIC AS
# MAGIC SELECT 
# MAGIC spine.*,
# MAGIC dups.hash1,
# MAGIC dups.hash2,
# MAGIC dups.hash3
# MAGIC FROM hash_df AS spine
# MAGIC INNER JOIN main.mosaic_end_to_end.temp_stage_hashes_dups AS dups ON dups.id = spine.id AND dups.DupRank = 1
# MAGIC WHERE 1=0
# MAGIC ;
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.mosaic_end_to_end.silver_dups_quarantine;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS main.mosaic_end_to_end.silver_dups_quarantine
# MAGIC AS
# MAGIC SELECT 
# MAGIC spine.*,
# MAGIC dups.hash1,
# MAGIC dups.hash2,
# MAGIC dups.hash3
# MAGIC FROM hash_df AS spine
# MAGIC INNER JOIN main.mosaic_end_to_end.temp_stage_hashes_dups AS dups ON dups.id = spine.id AND dups.DupRank > 1
# MAGIC WHERE 1=0
# MAGIC ;
# MAGIC
# MAGIC -- Upsert only non existing non-similar prompts
# MAGIC
# MAGIC MERGE INTO main.mosaic_end_to_end.silver_de_duped_hashed AS target
# MAGIC USING (
# MAGIC   SELECT 
# MAGIC spine.*,
# MAGIC dups.hash1,
# MAGIC dups.hash2,
# MAGIC dups.hash3
# MAGIC FROM hash_df AS spine
# MAGIC INNER JOIN main.mosaic_end_to_end.temp_stage_hashes_dups AS dups ON dups.id = spine.id AND dups.DupRank = 1
# MAGIC ) AS source
# MAGIC ON source.id = target.id
# MAGIC WHEN NOT MATCHED THEN INSERT *;
# MAGIC
# MAGIC
# MAGIC
# MAGIC MERGE INTO main.mosaic_end_to_end.silver_dups_quarantine AS target
# MAGIC USING (
# MAGIC   SELECT 
# MAGIC spine.*,
# MAGIC dups.hash1,
# MAGIC dups.hash2,
# MAGIC dups.hash3
# MAGIC FROM hash_df AS spine
# MAGIC INNER JOIN main.mosaic_end_to_end.temp_stage_hashes_dups AS dups ON dups.id = spine.id AND dups.DupRank > 1
# MAGIC ) AS source
# MAGIC ON source.id = target.id
# MAGIC WHEN NOT MATCHED THEN INSERT *;
# MAGIC
# MAGIC
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.mosaic_end_to_end.temp_stage_hashes_dups;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Check out final table
# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM main.mosaic_end_to_end.silver_de_duped_hashed

# COMMAND ----------

# DBTITLE 1,Create View for final training data set
# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE VIEW main.mosaic_end_to_end.gold_training_final_set
# MAGIC AS 
# MAGIC SELECT 
# MAGIC prompt,
# MAGIC response,
# MAGIC source
# MAGIC FROM main.mosaic_end_to_end.silver_de_duped_hashed;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC OPTIMIZE main.mosaic_end_to_end.silver_de_duped_hashed