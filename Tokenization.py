tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_input_length =128
max_target_length = 128
source_lang = "en"
target_lang = "hi"
def preprocess_function(examples):
  inputs = [ex[source_lang] for ex in examples['translation']]
  targets = [ex[target_lang] for ex in examples['translation']]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation = True)

  with tokenizer.as_target_tokenizer():
    labels = tokenizer(targets, max_length=max_target_length, truncation = True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs
  tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
  #extracting model
  model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
