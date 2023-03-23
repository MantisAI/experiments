# Data LLM

This repository holds experiments on using LLMS to prototype a solution

## Zero shot LLM

Predict

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

Evaluate

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