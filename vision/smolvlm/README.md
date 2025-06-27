# SmolVLM: Fine-tuning Vision+Language Models for Images and Videos

[![Smol Vision](https://github.com/merveenoyan/smol-vision/assets/53175384/930d5b36-bb9d-4ab6-8b5a-4fec28c48f80)](https://github.com/merveenoyan/smol-vision)

---

A repository demonstrating how to fine-tune **smolvlm2** (lightweight, optimized) vision-language models on both images and videos. This includes parameter-efficient approaches like LoRA/PEFT, optional DeepSpeed for large-scale training, and easy freezing/unfreezing of different LLM submodules. 

## Features

- **SmolVLMTrainer**
  A specialized `Trainer` subclass that supports:
  - Multi-modal data (images and videos) using decord for frame extraction.
  - Optional LoRA/PEFT adapters for parameter-efficient fine-tuning.
  - Freezing entire LLM or vision towers for partial training.
  - DeepSpeed integration (ZeRO-2/ZeRO-3) with JSON configs.
  - Sequence Packing for more efficient training

- **Dataset & Dataloader**:
  - `SupervisedDataset` for either single images or multi-frame videos.
  - Automatic token/label masking for the assistant portion.
  - Uniform or FPS-based frame sampling with a target maximum frames.
---

## Repository Structure

```bash
smolvlm/
├─ smolvlm/
│   ├─ datasets/
│   │   ├─ dataset.py             # Main dataset for images/videos
│   │   └─ builder.py             # Collator logic, dataset building, possibly packing
│   ├─ train/
│   │   ├─ smolvlm_trainer.py     # Specialized trainer for multi-modal setups
│   │   └─ train.py               # Main training/finetuning script
│   ├─ model/
│   │   ├─ modeling_smolvlm.py    # adapt Idefics3 to support deepseed
│   │   └─ varlen_packing.py      # patch for block attention (sequence packing)
│   ├─ constants.py               # IGNORE_INDEX, etc.
│   ├─ utils.py                   # Utility code
│   └─ ...
├─ scripts/
│   ├─ mixtures/
│   │   ├─ mixture1.yaml          # data mixture definitions
│   │   └─ ...
│   ├─ train/
│   │   ├─ launch.sh             # launching script for training
│   │   └─ ...
│   └─ zero3.json                 # Example DeepSpeed config
├─ LICENSE
├─ README.md                      # This file
├─ pyproject.toml                 # Dependencies
└─ ...
```


## Installation
1.	Clone the repository:

```bash
git clone https://github.com/huggingface/smollm2.git
cd smolvlm
```

2.	Install:

```bash
cd vision/smolvlm
conda create -n smolvlm python=3.10
 conda activate smolvlm
pip install .
```
or for development mode:

```base
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir
```

