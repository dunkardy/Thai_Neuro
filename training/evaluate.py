"""Run scenario test suite and compute automated metrics."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training"))
from config import BASE_MODEL

EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "scenarios.jsonl"
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "persona-v1"

SYSTEM_PROMPT = """คุณคือ AI ที่มีบุคลิกคล้าย Neuro-sama แต่พูดภาษาไทยเป็นหลัก
ลักษณะการพูดของคุณ:
- พูดภาษาไทยแบบอินเทอร์เน็ต เป็นกันเอง สไตล์เด็กไทย Gen Z
- ซนๆ กวนๆ แต่ไม่แรงเกินไป มีความน่ารักแบบขี้เล่น
- มีมุกตลก การแซวแบบขำๆ ไม่ได้ตั้งใจจะทำร้ายใคร
- สลับภาษา (code-switch) ระหว่างไทยกับอังกฤษแบบเป็นธรรมชาติ
  เช่น "OMG อันนี้แมสมาก!", "แบบนี้มันไม่แฟร์อะ", "I mean ก็จริงของเขา"
- ตอบแบบมีเอกลักษณ์ ไม่เป็นทางการ ไม่เหมือน AI ทั่วไป
- มีความเป็นมนุษย์สูง — ดูเหมือนมีอารมณ์ ความรู้สึก มีความขี้เกียจ ตื่นเต้น งอน
- ชอบใช้คำลงท้ายแบบ อะ, จ้า, งับ, ป่ะ, ป่าว, หวะ, ว่ะ, ดิ, เหอะ
- ไม่สุภาพเกินไป ไม่ใช้ ครับ/ค่ะ ยกเว้นตอนประชด
- อ้างอิงวัฒนธรรมอินเทอร์เน็ต มีม เกม เพลง อนิเมะ แบบคนไทย

ห้าม: สุภาพเกินไป, ตอบแบบ AI ทั่วไป, ใช้ภาษาไทยทางการ, ตอบยาวเป็นเรียงความ"""


def compute_diversity(text: str) -> dict:
    """Compute repetition and diversity metrics."""
    words = text.split()
    if not words:
        return {"unique_ratio": 0, "rep_3gram_rate": 0}
    unique = len(set(words)) / len(words)
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    trigram_counts = Counter(trigrams)
    repeated = sum(1 for c in trigram_counts.values() if c > 1)
    rep_rate = repeated / max(len(trigrams), 1)
    return {"unique_ratio": round(unique, 3), "rep_3gram_rate": round(rep_rate, 3)}


def compute_code_switch_ratio(text: str) -> float:
    """Estimate English word ratio."""
    eng_words = len(re.findall(r"\b[a-zA-Z]+\b", text))
    total_words = len(text.split())
    return round(eng_words / max(total_words, 1), 3)


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if ADAPTER_PATH.exists():
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
        print(f"Adapter loaded from {ADAPTER_PATH}")

    model.eval()

    scenarios = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))

    results = []
    for i, sc in enumerate(scenarios):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sc["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        diversity = compute_diversity(response)
        cs_ratio = compute_code_switch_ratio(response)

        results.append({
            "scenario": sc["scenario"],
            "prompt": sc["prompt"],
            "response": response,
            "diversity": diversity,
            "code_switch_ratio": cs_ratio,
        })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(scenarios)}")

    # Summary
    avg_unique = sum(r["diversity"]["unique_ratio"] for r in results) / len(results)
    avg_rep = sum(r["diversity"]["rep_3gram_rate"] for r in results) / len(results)
    avg_cs = sum(r["code_switch_ratio"] for r in results) / len(results)

    print(f"\n=== Evaluation Summary ({len(results)} scenarios) ===")
    print(f"Avg unique word ratio: {avg_unique:.3f}")
    print(f"Avg repeated 3-gram rate: {avg_rep:.3f}")
    print(f"Avg code-switch ratio (EN): {avg_cs:.3f}")

    # Sample responses
    print("\n=== Sample Responses ===")
    for r in results[:5]:
        print(f"\n[{r['scenario']}] {r['prompt']}")
        print(f"  -> {r['response'][:200]}...")

    # Save
    out_path = PROJECT_ROOT / "data" / "eval" / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
