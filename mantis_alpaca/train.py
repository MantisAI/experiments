from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from typing import Any, Dict, List, Tuple, Union
import typer

END_KEY = "### End"
INSTRUCTION_KEY = "### Instruction:"
INSTRUCTION_END = f"### Response:"


class DataCollatorForInstructionFineTuning(DataCollatorForLanguageModeling):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        batch = super().torch_call(examples)
        instruction_end_pattern = self.tokenizer.encode(INSTRUCTION_END)
        #print("INSTRUCTION END PATTERN ", instruction_end_pattern)

        for i in range(len(examples)):
            instruction_end_index = self.find_endpattern(instruction_end_pattern[0], batch["labels"][i])
            if instruction_end_index != -1:
                # Make sure that the instruction is ignored by the pytorch loss function
                batch["labels"][i, :instruction_end_index] = -100
            else:
                raise RuntimeError("Instruction end pattern not found in labels")
            
        return batch

    def find_endpattern(self, pattern, array):
        for i in range(len(array)):
            if array[i] == pattern:
                return i
        #pattern_length = len(pattern)
        #for i in range(len(array) - pattern_length + 1):
        #    if array[i : i + pattern_length] == pattern:
        #        return i
        return -1
            

def append_end_key(item):
    item["text"] += f"\n\n{END_KEY}"
    return item

def train(
    data_path="tatsu-lab/alpaca",
    model_path="alpaca-pythia-test",
    pretrained_model="EleutherAI/pythia-70m",
    learning_rate: float = 2e-5,
    batch_size: int = 1,
    weight_decay: float = 0.01,
    fp16: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    test_size: float = 0.1,
    epochs: int = 1,
    max_length: int = 1024,
    deepspeed_config: str = "deepspeed_config.json",
    local_rank: int = typer.Option("-1", "--local_rank")
):
    dataset = load_dataset(data_path)
    dataset = dataset.filter(lambda x: not x["text"].strip().endswith("### Response:"))
    dataset.map(append_end_key)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, INSTRUCTION_END]})

    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], max_length=max_length, truncation=True),
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )
    dataset = dataset["train"].train_test_split(test_size=test_size)

    model = AutoModelForCausalLM.from_pretrained(pretrained_model, use_cache=False)

    data_collator = DataCollatorForInstructionFineTuning(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        fp16=fp16,
        bf16=bf16,
        output_dir=model_path,
        evaluation_strategy="steps",
        eval_steps=100,
        num_train_epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_strategy="steps",
        logging_steps=10,
        weight_decay=weight_decay,
        report_to="wandb",
        deepspeed=deepspeed_config,
        local_rank=local_rank
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    typer.run(train)
