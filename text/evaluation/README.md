# SmolLM3 evaluation scripts

We're using the [LightEval](https://github.com/huggingface/lighteval/) library to benchmark our models.

## Setup

Use conda/uv/venv with `python>=3.11`.

For reproducibility, we recommend fixed versions of the libraries:
```bash
pip install uv
uv venv smol3_venv --python 3.11 
source smol3_venv/bin/activate

GIT_LFS_SKIP_SMUDGE=1 uv pip install -r requirements.txt
```

## Running the evaluations

### SmolLM3 base models

```bash
lighteval vllm \
    "model_name=HuggingFaceTB/SmolLM3-3B-Base,dtype=bfloat16,max_model_length=32768,max_num_batched_tokens=32768,generation_parameters={temperature:0},tensor_parallel_size=2,gpu_memory_utilization=0.7" \
    "smollm3_base.txt" \
    --custom-tasks "tasks.py" \
    --output-dir "evals/" \
    --save-details
```

### SmolLM3 instruction-tuned models

(note the `--use_chat_template` flag)
```bash
TODO
```

