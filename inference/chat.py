"""Interactive CLI chat for testing the fine-tuned persona model."""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

PROJECT_ROOT = Path(__file__).parent.parent
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "persona-v1"
BASE_MODEL = "TrevorJS/gemma-4-E4B-it-uncensored"

SYSTEM_PROMPT = (
    "คุณคือ AI ที่มีบุคลิกขี้เล่น กวนๆ แบบ Neuro-sama แต่พูดภาษาไทย "
    "พูดแบบคนไทย Gen Z มี code-switching เป็นธรรมชาติ "
    "ไม่ต้องสุภาพ ไม่ต้องตอบแบบ AI ทั่วไป"
)


def main():
    print("Loading model (4-bit with adapter)...")
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if ADAPTER_PATH.exists():
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
        print(f"Adapter loaded from {ADAPTER_PATH}")
    else:
        print("No adapter found, using base model only")

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n=== Neuro-Thai Chat ===\nType 'exit' to quit, 'clear' to reset\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Chat cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        print("\nBot: ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.9,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                streamer=streamer,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": response})
        print()


if __name__ == "__main__":
    main()
