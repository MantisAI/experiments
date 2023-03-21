from pathlib import Path

from datasets import load_dataset
import srsly
import typer


def get_data(dataset_name, data_dir: Path = "data"):
    dataset = load_dataset(dataset_name)

    data_dir.mkdir(parents=True, exist_ok=True)

    id2label = {
        i: label for i, label in enumerate(dataset["train"].features["label"].names)
    }
    for split in ["train", "test"]:
        data = [
            {"text": example["text"], "label": id2label[example["label"]]}
            for example in dataset[split]
        ]
        srsly.write_jsonl(data_dir / f"{split}.jsonl", data)


if __name__ == "__main__":
    typer.run(get_data)
