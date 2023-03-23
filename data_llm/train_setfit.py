from pathlib import Path

from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from datasets import Dataset
import srsly
import typer


def train(
    data_path,
    test_data_path,
    result_path: Path,
    model_name="sentence-transformers/paraphrase-mpnet-base-v2",
):
    data = list(srsly.read_jsonl(data_path))
    test_data = list(srsly.read_jsonl(test_data_path))

    labels = list(set([example["label"] for example in data]))
    label2id = {label: i for i, label in enumerate(labels)}
    data = [
        {"text": example["text"], "label": label2id[example["label"]]}
        for example in data
    ]
    test_data = [
        {"text": example["text"], "label": label2id[example["label"]]}
        for example in test_data
    ]

    train_dataset = Dataset.from_list(data)
    train_dataset = sample_dataset(train_dataset, label_column="label", num_samples=8)
    test_dataset = Dataset.from_list(test_data)

    model = SetFitModel.from_pretrained(model_name)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=16,
        num_iterations=20,
        num_epochs=1,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(metrics)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    srsly.write_json(result_path, metrics)


if __name__ == "__main__":
    typer.run(train)
