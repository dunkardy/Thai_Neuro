"""Fine-tune Gemma 4 E4B (abliterated) with QLoRA using Unsloth."""
import json
from pathlib import Path

from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# Import config relative to this file
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BASE_MODEL,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    MAX_SEQ_LENGTH,
    PER_DEVICE_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    NUM_EPOCHS,
    LOAD_IN_4BIT,
    PROCESSED_DIR,
    OUTPUT_DIR,
)


def load_conversations(path: Path) -> list[dict]:
    """Load processed conversations from JSONL."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_for_training(conversations: list[dict], tokenizer) -> list:
    """Apply chat template to each conversation."""
    formatted = []
    for conv in conversations:
        try:
            text = tokenizer.apply_chat_template(
                conv["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted.append({"text": text})
        except Exception:
            pass
    return formatted


def main():
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Apply QLoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load and format data
    train_data = load_conversations(PROCESSED_DIR / "train.jsonl")
    val_data = load_conversations(PROCESSED_DIR / "val.jsonl")

    if not train_data:
        print("No training data found. Run prepare_data.py first.")
        return

    train_formatted = format_for_training(train_data, tokenizer)
    val_formatted = format_for_training(val_data, tokenizer) if val_data else []

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted) if val_formatted else None

    print(f"Train samples: {len(train_dataset)}", end="")
    if val_dataset:
        print(f", Val samples: {len(val_dataset)}")
    else:
        print()

    # Train
    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=100,
            save_steps=500,
            output_dir=str(OUTPUT_DIR),
            report_to="none",
        ),
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
