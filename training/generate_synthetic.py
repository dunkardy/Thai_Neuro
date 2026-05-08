"""Generate persona-consistent Thai conversations via DeepSeek API."""
import json
import os
import time
from pathlib import Path

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import SYNTHETIC_MODEL

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"

# CANONICAL PERSONA PROMPT — keep in sync with inference/chat.py and training/evaluate.py
PERSONA_SYSTEM = """คุณคือ AI ที่มีบุคลิกคล้าย Neuro-sama แต่พูดภาษาไทยเป็นหลัก
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

SCENARIOS = [
    "casual_chat",
    "roast_playful",
    "absurd_humor",
    "emotional",
    "gaming_trash_talk",
    "existential",
    "reaction",
    "debate_playful",
    "daily_life",
    "music_entertainment",
]


def generate_conversation(
    client: OpenAI,
    scenario: str,
    num_turns: int = 4,
) -> dict | None:
    """Generate a multi-turn conversation for a given scenario."""
    prompt = f"""สร้างบทสนทนาภาษาไทยแบบ multi-turn ระหว่าง User (คนไทยทั่วไป)
กับ Assistant (AI ที่มีบุคลิกตาม system prompt)
สถานการณ์: {scenario}
จำนวน: {num_turns} turns

ให้ตอบกลับเป็น JSON Array ในรูปแบบ:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]

แต่ละ turn ต้องเป็นธรรมชาติ มี code-switching เหมือนคนไทยจริง
Assistant ต้องตอบแบบมีเอกลักษณ์ตามบุคลิก ไม่ใช่ AI ทั่วไป"""

    try:
        resp = client.chat.completions.create(
            model=SYNTHETIC_MODEL,
            max_tokens=2048,
            temperature=0.9,
            messages=[
                {"role": "system", "content": PERSONA_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content
        if not content:
            return None
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            return None
        turns = json.loads(content[start:end])
        return {"messages": turns, "scenario": scenario}
    except Exception as e:
        print(f"  Error generating {scenario}: {e}")
        return None


def _preview(content: str, max_len: int = 80) -> str:
    """First line or truncated snippet of an assistant reply."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:max_len] + ("..." if len(stripped) > max_len else "")
    return "(empty)"


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY not set")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUTPUT_DIR / "synthetic_conversations.jsonl"
    num_per_scenario = 50
    total = 0

    print(f"Using model: {SYNTHETIC_MODEL}")
    with open(out_path, "w", encoding="utf-8") as f:
        for scenario in SCENARIOS:
            print(f"\n[{scenario}]")
            for i in range(num_per_scenario):
                conv = generate_conversation(client, scenario, num_turns=4)
                if conv:
                    first_assistant = next(
                        (m["content"] for m in conv["messages"] if m["role"] == "assistant"), ""
                    )
                    preview = _preview(first_assistant)
                    print(f"  [{i + 1}/{num_per_scenario}] {preview}")
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    f.flush()
                    total += 1
                else:
                    print(f"  [{i + 1}/{num_per_scenario}] FAILED")
                time.sleep(0.3)

    print(f"\nSaved {total} conversations to {out_path}")


if __name__ == "__main__":
    main()
