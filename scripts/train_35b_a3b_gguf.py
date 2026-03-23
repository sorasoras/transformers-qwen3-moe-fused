#!/usr/bin/env python3
"""
Train script for models intended to be exported to GGUF (e.g. 35b_a3b).
This mirrors the Qwen-3.5 training script but adds an optional post-training GGUF export helper (user-provided converter).

Usage (example):
python scripts/train_35b_a3b_gguf.py \
  --model_name_or_path qwen/qwen-3.5-35b-a3b \
  --output_dir outputs/35b_a3b_lora \
  --train_file data/train.jsonl \
  --do_train \
  --apply_lora \
  --export_gguf "convert-command --from huggingface --model outputs/35b_a3b_lora --to outputs/35b_a3b_lora.gguf"
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
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--export_gguf", type=str, default=None, help="Optional shell command to export trained model to GGUF. Enclose in quotes.")
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

    if args.use_4bit and prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model)

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

        if args.export_gguf:
            logger.info("Running GGUF export command provided by user...")
            rc = os.system(args.export_gguf)
            if rc != 0:
                logger.warning(f"Export command exited with code {rc}. Please check the command and run manually if needed.")
            else:
                logger.info("Export command finished.")

if __name__ == "__main__":
    main()
