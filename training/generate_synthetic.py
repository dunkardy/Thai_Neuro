"""Generate persona-consistent Thai conversations via Claude/GPT-4o API."""
import json
import os
import time
from pathlib import Path

from anthropic import Anthropic

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
    client: Anthropic,
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
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=PERSONA_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.content[0].text
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            return None
        turns = json.loads(content[start:end])
        return {"messages": turns, "scenario": scenario}
    except Exception as e:
        print(f"  Error generating {scenario}: {e}")
        return None


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set")
        return

    client = Anthropic(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_conversations = []
    num_per_scenario = 50

    for scenario in SCENARIOS:
        print(f"Generating {scenario}...")
        for i in range(num_per_scenario):
            conv = generate_conversation(client, scenario, num_turns=4)
            if conv:
                all_conversations.append(conv)
            time.sleep(0.3)
        print(f"  Generated {num_per_scenario} for {scenario}")

    out_path = OUTPUT_DIR / "synthetic_conversations.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_conversations)} conversations to {out_path}")


if __name__ == "__main__":
    main()
