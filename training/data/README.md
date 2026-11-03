# Ogun Training Data & Model Artifacts

This directory holds dataset caches, normalized DPO pairs, tokenized data, and model export artifacts for the Ogun local training pipeline.

## Directory Structure:
- `raw/`: Raw downloaded datasets and chunks (cached from Hugging Face).
- `merged_dpo_dataset.jsonl`: Cleaned, filtered, and normalized DPO preference dataset pairs.
- `merged_dpo_dataset.stats.json`: Ingestion and drop statistics, token/character length percentiles.
- `tokenized/`: Intermediate tokenized datasets (if cached ahead of training).
- `exports/<run-name>/`: Saved 4-bit LoRA adapter weights, checkpoints, merged weights, and exported GGUF binaries.
