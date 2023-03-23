from pathlib import Path
import os

import seaborn as sns
import srsly
import typer


def plot(results_dir, figure_path: Path):
    techniques = []
    accuracies = []
    for result_file in os.listdir(results_dir):
        technique = os.path.splitext(result_file)[0]
        result = srsly.read_json(os.path.join(results_dir, result_file))
        accuracy = result["accuracy"]

        techniques.append(technique)
        accuracies.append(accuracy)

    plt = sns.barplot(x=techniques, y=accuracies, palette="vlag")
    plt.set_title("Accuracy by technique")
    fig = plt.get_figure()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)


if __name__ == "__main__":
    typer.run(plot)
