Gemma 3: Custom LLM with LoRA Fine-Tuning
Overview
This repository contains an implementation of Gemma 3, a transformer-based language model, along with LoRA-based fine-tuning on a small dataset. The model is built from scratch with modular components including attention, feedforward blocks, and custom RMS normalization.

Gemma 3 is designed to support both sliding-window local attention and full global attention. It uses:

RMSNorm for normalization (similar to HuggingFace Gemma3 implementation)
Grouped Query Attention (GQA) with optional RoPE embeddings
Two-expansion feedforward blocks with GELU activation
Modular Transformer blocks with skip connections
Token embeddings and output projection head
Pre-training Dataset
from datasets import load_dataset

# TinyStories dataset
ds = load_dataset("roneneldan/TinyStories")
LoRA Fine-Tuning
LoRA (Low-Rank Adaptation) was used to fine-tune the pre-trained Gemma 3 model on instruction dataset

Pre-Processing of the Fine-Tuning Dataset
Formatted the dataset in Alpaca style for instruction-response training.
Implemented a custom collate function to combine data and enforce a maximum token limit per batch.
Key points:
Fine-tuning was performed from scratch using ~1000 rows of dataset.
Only LoRA weights were updated, keeping the majority of pre-trained weights frozen.
Training used cross-entropy loss with batch size, optimizer, and learning rate chosen for small-scale fine-tuning.
The model supports both pretraining and generation functions.
Pre-Training Results
Example generation after pretraining on TinyStories:

Input: "Once upon a time there was a pumpkin."
Output: "Once upon a time there was a pumpkin. It was who wanted to see his heart also sad. Todd right bubbles with a Anna and juicy! Once upon a time there was a sail. The house and’s dance dog..."
Observations:
The model produces coherent sentences initially, but quickly degrades into repetitive and semi-nonsensical text.
This is expected because the pre-training dataset is very small (~1k samples) and the model has millions of parameters.
LoRA Fine-Tuning Results
Example instruction-response after LoRA fine-tuning:

Instruction: Rewrite the sentence using a simile.
Input: The car is very fast.
Model Response: The moral of the word, the little girl is ' is ' is ' is ' is ' is 'I'm sorry.

Instruction: What type of cloud is typically associated with thunderstorms?
Model Response: The moral of ' is ' is 'I'm sorry.


Instruction: Name the author of 'Pride and Prejudice'.
Model Response: The moral of ' is 'You're welcome, and 'You're welcome...
Observations:
The model fails to follow instructions and outputs repetitive patterns like 'The moral of...' or 'I'm sorry'.
This is because the LoRA fine-tuning dataset is extremely small and pre-training is done on a story telling dataset so the model did not learn proper instruction-response mapping.
LoRA only updates low-rank adapters, so with insufficient data, the model cannot generalize well for new instructions.
Why Results Are Like This
Small Dataset: ~tinystories dataset for a 270M-parameter model to reliably learn structured text generation.
LoRA Limited Capacity: Only a fraction of the model's parameters are updated, so fine-tuning is extremely data-dependent.
Instruction Complexity: The model has not seen enough varied instructions to generalize properly.
Pretraining Influence: Initial pretraining generates mostly story-like text; without enough instruction-specific fine-tuning, the model defaults to learned patterns (hence repetitive 'The moral of...' outputs).
Future Improvements
Increase dataset size for instruction-following tasks (several thousand examples or more).
Combine LoRA fine-tuning with full model fine-tuning if resources allow.
Incorporate diverse instruction datasets to improve generalization.
Experiment with temperature, top-k/top-p sampling during generation to reduce repetitive outputs.
