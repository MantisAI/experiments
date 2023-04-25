# Mantis meets Alpaca

We want to experiment with fine tuning an LLM on top of the Alpaca
dataset here

## Contribute

This project is using `poetry` to manage dependencies and virtualenvs. You can run a script using
`poetry run` and add dependencies using `poetry add`. Read more about Poetry [here](https://python-poetry.org/)


## Setting up p4d.xlarge instance for training

`nvidia-fabricmanager` is needed to make cuda work properly on instances that use A100 GPUs. `datacenter-gpu-manager` installs proper drivers also (finds that it's an aws instance and selects proper drivers, nice). Steps to install:

```
sudo apt-get install -y datacenter-gpu-manager
sudo systemctl --now enable nvidia-dcgm

sudo apt-get install cuda-drivers-fabricmanager
sudo systemctl start nvidia-fabricmanager
``` 

Then install `torch` normally, no need to use specific cuda version as of late.

## Estimating Memory requirements for models

Deepspeed provides some functions to estimate a few possible scenarios of memory consumption of GPU and RAM (reffered as CPU by deepspeed).
While it can be estimated for all stages, for training we only care about the lates stage 3. Example:

```
from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
model = AutoModel.from_pretrained("t5-3b")
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
```

gives output:

```
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 8 GPUs per node.
SW: Model with 2851M total params, 32M largest layer params.
  per CPU  |  per GPU |   Options
   71.71GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
  127.48GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   63.74GB |   0.79GB | offload_param=none, offload_optimizer=cpu , zero_init=1
  127.48GB |   0.79GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    1.47GB |   6.10GB | offload_param=none, offload_optimizer=none, zero_init=1
  127.48GB |   6.10GB | offload_param=none, offload_optimizer=none, zero_init=0
```

Mentioning that also you can estimate this on local machine without downloading the model, all you need to know is the aprox number of parameters and ideally also the number of parameters in the largest layer. 
So for the example above (`t5-3b`) I can call a `cold` version of the function, that doesn't need the actual model:

```
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_cold
estimate_zero3_model_states_mem_needs_all_cold(total_params=2851e6, largest_layer_params=32e6, num_gpus_per_node=8, num_nodes=1)
```

gives output:

```
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 8 GPUs per node.
SW: Model with 2851M total params, 32M largest layer params.
  per CPU  |  per GPU |   Options
   71.69GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
  127.45GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   63.72GB |   0.78GB | offload_param=none, offload_optimizer=cpu , zero_init=1
  127.45GB |   0.78GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    1.43GB |   6.09GB | offload_param=none, offload_optimizer=none, zero_init=1
  127.45GB |   6.09GB | offload_param=none, offload_optimizer=none, zero_init=0
```

`offload_optimizer=cpu` means we're offloading memory required by gradients and other training components to cpu
`offload_param=cpu` means that we'll oflload the model weights to the CPU as well. So if this remains `none` that means that is also the memory needed for inference.
`zero_init` means Deepspeed zero stage (3 in this case) enabled or not
