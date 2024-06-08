tokenized = tokenizer([input_text], return_tensors='np')
out = model.generate(**tokenized, max_length=128)
print(out)
with tokenizer.as_target_tokenizer():
    print(tokenizer.decode(out[0], skip_special_tokens=True))
def generate_translation(input_text):
  tokenized = tokenizer([input_text], return_tensors='np')
  out = model.generate(**tokenized, max_length=128)
  with tokenizer.as_target_tokenizer():
    print(tokenizer.decode(out[0], skip_special_tokens=True))
