optimizer=AdamWeightDecay(learning_rate=learning_rate,weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer)
model.fit(train_dataset,validation_data=validation_dataset,epochs=num_train_epochs)
model.save_pretrained("tf_model/")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model/")