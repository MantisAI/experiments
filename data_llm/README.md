# Data LLM 🤖

In this experiment we are exploring different ways we can use a Large Language Model
like GPT3 to prototype a solution. In particular we investigate

- using the LLM as a zero shot model in the loop
- creating synthetic data and training a model on that data
- labeling an unlabeled dataset and training a model on that data

We use [ag_news](https://huggingface.co/datasets/ag_news) as our dataset to experiment on 
and [SetFit](https://github.com/huggingface/setfit) to train models

## Quickstart

Install the requirements, preferablly in an isolated environment
```
pip install -r requirements.txt
```

🦉 Run `dvc repro` to reproduce our results or use / modify the relevant scripts
described below

## Data

We download the data from the huggingface hub using the `get_data` script.

```
 Usage: get_data.py [OPTIONS] DATASET_NAME                                                                                                                  
                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    dataset_name      TEXT  [default: None] [required]                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --data-dir        PATH  [default: data]                                                                                                                  │
│ --help                  Show this message and exit.                                                                                                      │
╰────────────────────────
```

## Zero shot LLM

We use the `zero_shot_llm` script to make predictions using GPT3. We use
the data generated to evaluate the accuracy of this approach but also
to train a smaller model using SetFit.

```
 Usage: zero_shot_llm.py predict [OPTIONS] DATA_PATH PRED_DATA_PATH                                                                                         
                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path           TEXT  [default: None] [required]                                                                                                │
│ *    pred_data_path      TEXT  [default: None] [required]                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --model              TEXT     [default: text-davinci-003]                                                                                                │
│ --sample-size        INTEGER  [default: 100]                                                                                                             │
│ --help                        Show this message and exit.                                                                                                │
╰─────────────────────────────────────
```

We evaluate the zero shot approach using the command below

```
 Usage: zero_shot_llm.py evaluate [OPTIONS] DATA_PATH PRED_DATA_PATH                                                                                        
                                  RESULT_PATH                                                                                                               
                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path           TEXT  [default: None] [required]                                                                                                │
│ *    pred_data_path      TEXT  [default: None] [required]                                                                                                │
│ *    result_path         PATH  [default: None] [required]                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --sample-size        INTEGER  [default: 100]                                                                                                             │
│ --help                        Show this message and exit.                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## Synthetic data

We create synthetic data using the script `create_synthetic_data`

```
 Usage: create_synthetic_data.py [OPTIONS] LABELS DATA_PATH                                                                                                 
                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    labels         TEXT  [default: None] [required]                                                                                                     │
│ *    data_path      TEXT  [default: None] [required]                                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --model                         TEXT     [default: text-davinci-003]                                                                                     │
│ --num-examples-per-label        INTEGER  [default: 8]                                                                                                    │
│ --help                                   Show this message and exit.                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## Setfit

We use `train_setfit` to train models for

- the synthetic data
- the data from zero shot llm
- the original train data

```
Usage: train_setfit.py [OPTIONS] DATA_PATH TEST_DATA_PATH RESULT_PATH                                                                                      
                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path           TEXT  [default: None] [required]                                                                                                │
│ *    test_data_path      TEXT  [default: None] [required]                                                                                                │
│ *    result_path         PATH  [default: None] [required]                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --model-name        TEXT  [default: sentence-transformers/paraphrase-mpnet-base-v2]                                                                      │
│ --help                    Show this message and exit.                                                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Plot results

Finally we plot all results collected in the results folder

```
Usage: plot_results.py [OPTIONS] RESULTS_DIR FIGURE_PATH                                                                                                   
                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    results_dir      TEXT  [default: None] [required]                                                                                                   │
│ *    figure_path      PATH  [default: None] [required]                                                                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```