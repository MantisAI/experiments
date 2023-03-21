import os

from tqdm import tqdm
import tiktoken
import openai
import srsly
import typer

MODELS_COST = {
    "text-davinci-003": 0.0200,
    "text-curie-001": 0.0020,
    "text-babbage-001": 0.0005,
    "text-ada-001": 0.0004,
}


def gpt(model, text, labels):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = f"Classify the text into one of the following labels: {', '.join(labels)}\n\nText: {text}\nLabel:"

    response = openai.Completion.create(
        model=model, prompt=prompt, temperature=0, max_tokens=7
    )
    return response["choices"][0]["text"].strip()


def predict(
    data_path, pred_data_path, model="text-davinci-003", sample_size: int = 100
):
    data = list(srsly.read_jsonl(data_path))

    if sample_size:
        data = data[:sample_size]

    labels = list(set([example["label"] for example in data]))

    encoding = tiktoken.encoding_for_model(model)
    num_tokens = sum([len(encoding.encode(example["text"])) for example in data])

    price = num_tokens * MODELS_COST[model] / 1000
    if typer.confirm(f"This will cost you {price:.2f}$. Do you want to continue?"):
        pred_data = []
        for example in tqdm(data):
            pred_label = gpt(model, example["text"], labels)
            pred_data.append({"text": example["text"], "label": pred_label})

        srsly.write_jsonl(pred_data_path, pred_data)


if __name__ == "__main__":
    typer.run(predict)
