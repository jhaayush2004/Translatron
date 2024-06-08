batch_size = 16
learning_rate =2e-5
weight_decay=0.01
num_train_epochs = 1
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,return_tensors="np")
train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["test"],
    batch_size = batch_size,
    shuffle = True,
    collate_fn = data_collator,
)
validation_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
)
generation_dataset = model.prepare_tf_dataset(
    tokenized_datasets["test"],
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator,
)
