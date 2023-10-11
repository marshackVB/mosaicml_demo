# Databricks notebook source
# MAGIC %md #### Load an MPT model fine tuned using MosaicML's finetune API. 
# MAGIC Developed on DBR ML 14.0. 
# MAGIC
# MAGIC Relevent docs:  
# MAGIC  - MPT-instruct [documentation](https://huggingface.co/mosaicml/mpt-7b-instruct#how-to-use)
# MAGIC  - Databricks [blog](https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html#:~:text=The%20MLflow%20transformers%20flavor%20supports,chatbot.) on the MLflow transformer model flavor

# COMMAND ----------

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, logging

# COMMAND ----------

logging.set_verbosity(50)

# COMMAND ----------

# MAGIC %md Load model

# COMMAND ----------

model_path = '/dbfs/Users/marshall.carter@databricks.com/mosaicml/finetune'

config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.init_device = 'cuda:0'

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             config = config,
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True
                                             )

tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

# COMMAND ----------

# MAGIC %md Create stopping criteria

# COMMAND ----------

stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

class StopOnTokens(StoppingCriteria):
  """
  Prevent model from generating extraneous text
  """
  def __call__(self, input_ids, scores, **kwargs):
    for stop_id in stop_token_ids:
      if input_ids[0][-1] == stop_id:
        return True
      return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# COMMAND ----------

# MAGIC %md Format prompt

# COMMAND ----------

INSTRUCTION_KEY = "### Instruction"
RESPONSE_KEY = "### Response"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro} {instruction_key} {instruction} {response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

example = "Which is the largest mammal in the world?"
fmt_ex = PROMPT_FOR_GENERATION_FORMAT.format(instruction=example)
print(fmt_ex)

# COMMAND ----------

# MAGIC %md Generate response

# COMMAND ----------

with torch.autocast('cuda', dtype=torch.bfloat16):
    generated_text = pipe(fmt_ex,
                          do_sample=True,
                          use_cache=True,
                          return_full_text=False, 
                          stopping_criteria=stopping_criteria,
                          num_return_sequences=1,
                          max_new_tokens=1000,
                          top_p=0.90, 
                          top_k=0,
                          temperature=0.1,
                          repetition_penalty = 1.1,
                          clean_up_tokenization_spaces=True)
    
print(generated_text[0]['generated_text'])
