#!/usr/bin/env python3
"""
Training script that integrates optional Unsloth optimizations for efficient training.
Supports bitsandbytes 4-bit loading, PEFT LoRA, and an optional local Unsloth integration.

Usage example:
python scripts/train_unsloth.py \
  --model_name_or_path qwen/qwen-3.5-35a3 \
  --output_dir outputs/qwen3.5-unsloth-lora \
  --train_file data/train.jsonl \
  --do_train \
  --apply_unsloth \
  --apply_lora \
  --use_4bit
"""
import argparse
import logging
import os
from pathlib import Path

import datasets
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Optional libraries: bitsandbytes, peft
try:
    import bitsandbytes as bnb  # noqa: F401
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    bnb = None
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

# Optional Unsloth integration from repo or external package
try:
    # Expected interface: apply_unsloth(model, config) -> model
    from unsloth import apply_unsloth  # noqa: F401
except Exception:
    apply_unsloth = None

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_file", required=True, help="JSONL or text dataset file")
    p.add_argument("--validation_file", default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--use_4bit", action="store_true", help="Load model in 4-bit via bitsandbytes")
    p.add_argument("--apply_lora", action="store_true", help="Apply LoRA (requires peft)")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--apply_unsloth", action="store_true", help="Apply Unsloth optimizations if available")
    p.add_argument("--unsloth_config", type=str, default=None, help="Path to Unsloth config JSON or stringified config")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return p.parse_args()

def load_dataset_train(file_path):
    ext = Path(file_path).suffix.lower()
    if ext in [".json", ".jsonl"]:
        ds = datasets.load_dataset("json", data_files={"train": file_path})
        return ds["train"]
    else:
        ds = datasets.load_dataset("text", data_files={"train": file_path})
        return ds["train"]

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Arguments: {args}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if args.use_4bit:
        if bnb is None:
            raise RuntimeError("bitsandbytes is required for --use_4bit but not available.")
        model_kwargs.update({"load_in_4bit": True, "device_map": "auto"})

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    # Optionally apply Unsloth optimizations
    if args.apply_unsloth:
        if apply_unsloth is None:
            logger.warning("Unsloth integration not available (unsloth.apply_unsloth not found). Skipping.)")
        else:
            logger.info("Applying Unsloth optimizations to model...")
            model = apply_unsloth(model, args.unsloth_config)

    # Prepare for k-bit training if using 4-bit
    if args.use_4bit and prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA if requested
    if args.apply_lora:
        if get_peft_model is None:
            raise RuntimeError("PEFT is required for --apply_lora but not available.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info("Applying LoRA config...")
        model = get_peft_model(model, lora_config)

    # Load dataset
    train_dataset = load_dataset_train(args.train_file)

    def preprocess(example):
        text = example.get("text") or example.get("content") or example.get("prompt") or example.get("input") or example.get("sentence") or example
        if isinstance(text, dict):
            text = text.get("text") or ""
        enc = tokenizer(text, truncation=True, max_length=args.max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", [1] * len(enc["input_ids"]))}

    tokenized = train_dataset.map(preprocess, remove_columns=train_dataset.column_names, batched=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=not args.use_4bit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized if args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()