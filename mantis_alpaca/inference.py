from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


model = AutoModelForCausalLM.from_pretrained(
  "alpaca-pythia-test/checkpoint-500/",
).to('cuda')

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1.4b",
)


prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Write a tweet about Mantis based on the next info:
We solve business problems related to natural human language and speech. This field of Artificial Intelligence is called Natural Language Processing (NLP).We use tools like spaCy, and transformers, which allow us to deploy state of the art models known for their speed and accuracy.

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# each of these is encoded to a single token
response_key_token_id = tokenizer.encode("### Response:")[0]
end_key_token_id = tokenizer.encode("### End")[0]

tokens = model.generate(
    input_ids, 
    pad_token_id=tokenizer.pad_token_id, 
    eos_token_id=end_key_token_id,
    do_sample=True, 
    max_new_tokens=64, 
    top_p=0.92, 
    top_k=50,
    )[0].cpu()

result = tokenizer.decode(tokens)
print(result)