### Databricks and MosaicML integrations demo

Files:  
**write_to_mds.py**: Convert a huggingface dataset to a Spark Dataframe; convert the Dataframe to MDS format. MDS files will then be streamed from S3 to the MosaicML compute plane during fine tuning.