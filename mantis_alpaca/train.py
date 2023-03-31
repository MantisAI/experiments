from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import typer


def train(
    data_path="tatsu-lab/alpaca",
    model_path="alpaca-pythia-test",
    pretrained_model="EleutherAI/pythia-70m",
    learning_rate: float = 2e-5,
    batch_size: int = 1,
    weight_decay: float = 0.01,
    test_size: float = 0.1,
    epochs: int = 3,
    max_length: int = 512,
):
    dataset = load_dataset(data_path)
    dataset = dataset.filter(lambda x: not x["text"].strip().endswith("### Response:"))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], max_length=max_length, truncation=True),
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )
    dataset = dataset["train"].train_test_split(test_size=test_size)

    model = AutoModelForCausalLM.from_pretrained(pretrained_model, device_map="auto")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        report_to="wandb",
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
