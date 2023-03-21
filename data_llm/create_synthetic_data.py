import os

from tqdm import tqdm
import openai
import srsly
import typer


def gpt(model, label, num_examples):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = f"Create {num_examples} one sentence news articles about {label}"

    response = openai.Completion.create(
        model=model, prompt=prompt, temperature=0, max_tokens=num_examples * 30
    )

    texts = [
        text.strip() for text in response["choices"][0]["text"].split("\n") if text
    ]

    return texts


def create_synthetic_data(
    labels, data_path, model="text-davinci-003", num_examples_per_label: int = 8
):
    data = []
    for label in tqdm(labels.split(",")):
        texts = gpt(model, label, num_examples_per_label)

        data.extend([{"text": text, "label": label} for text in texts])

    srsly.write_jsonl(data_path, data)


if __name__ == "__main__":
    typer.run(create_synthetic_data)
