# SmolVLM: Fine-tuning Vision+Language Models for Images and Videos

[![Smol Vision](https://github.com/merveenoyan/smol-vision/assets/53175384/930d5b36-bb9d-4ab6-8b5a-4fec28c48f80)](https://github.com/merveenoyan/smol-vision)

---

A repository demonstrating how to fine-tune **smol** (lightweight, optimized) vision-language models on both images and videos. This includes parameter-efficient approaches like LoRA/PEFT, optional DeepSpeed for large-scale training, and easy freezing/unfreezing of different LLM submodules. 

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
git clone https://github.com/your-org/smolvlm.git
cd smolvlm
```

2.	Install:

```bash
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

## what was added?
* support for images with different shapes
* support for mixed batches
* sequence packing
* deepspeed integration
* yaml dataset configs with dataset subsampling
* gradient checkpointing
* different lr for vision, connector, and llm
* support multi-image data

# TODO
* Fix the squence packing logic. Need 'trainer' to update training steps re. packing.
* right now, you use `<image>` for tokens. Note that this is common in HTML. Better to use `<|image_token|>`. Potentially, we can seperate image and video tokens, `<|video_token|>`. 
* support adding new tokens (need to unfreeze lm head, resize tokenizer)
* Improve tokenizer logic. what we have right now is very breakable. if we are re-training smolVLM, lets improve tokenization stratagey as well.
* Add Liger Kernel for faster training
* Add RingAttention for long-sequence training
* For video: average the embeddings of N(=4 or 8) images. 

# Questions:
* Current 'image-splitting' logic is to add tokens, e.g., `'<row_2_col_1>'`. However, the tokenizer does not have these tokens. they are not being encoded properly (i.e., there are no learned tokens to represent each of the positions in the image. this does not happen.)
* right now, i batch images/videos etc of different sizes by adding `pixel_attention_mask`. Verify that this is what was done in smolVLM.
* You set the largest image size to 1920. However, the code shows this should be 1536. why did you use 1920? are you certain this is desired? 

# Suggestions:
* We can use the [PaliGemma 896x encoder](https://huggingface.co/google/paligemma2-28b-pt-896), and either do no or minipal patching (2x2). Alternatively, we can use the [448x](https://huggingface.co/google/paligemma2-28b-pt-448) varient with the same image splitting logic and support larger images. 
* Potentially, we should be using the SigLIP-B-512. I wonder what the performance difference will be, and SigLIP-SO400M has 878M params vs the Base 204M. This way, you could use also the smaller LLMs