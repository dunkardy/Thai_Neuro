"""Merge scraped + synthetic data, clean, format, and split into train/val."""
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRAPED_DIR = PROJECT_ROOT / "data" / "scraped"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MIN_MESSAGE_LENGTH = 10
MAX_MESSAGE_LENGTH = 2048
TRAIN_RATIO = 0.9

PHONE_PATTERN = re.compile(r"0\d{1,2}[-\s]?\d{3}[-\s]?\d{4}")
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
URL_PATTERN = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """Remove PII and normalize whitespace."""
    text = PHONE_PATTERN.sub("[เบอร์โทร]", text)
    text = EMAIL_PATTERN.sub("[อีเมล]", text)
    text = URL_PATTERN.sub("[ลิงก์]", text)
    text = " ".join(text.split())
    return text


def deduplicate(conversations: list[dict]) -> list[dict]:
    """Remove near-duplicate conversation pairs."""
    seen = set()
    unique = []
    for conv in conversations:
        key = json.dumps(conv["messages"], ensure_ascii=False)[:200]
        if key not in seen:
            seen.add(key)
            unique.append(conv)
    return unique


def is_valid(conv: dict) -> bool:
    """Filter low-quality conversations."""
    for msg in conv.get("messages", []):
        text = msg.get("content", "")
        if len(text) < MIN_MESSAGE_LENGTH:
            return False
        if len(text) > MAX_MESSAGE_LENGTH:
            return False
    return len(conv["messages"]) >= 2


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, return list of dicts."""
    if not path.exists():
        print(f"  {path} not found, skipping")
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all sources
    all_conversations = []
    sources = [
        SCRAPED_DIR / "pantip" / "conversations.jsonl",
        SCRAPED_DIR / "reddit" / "conversations.jsonl",
        SCRAPED_DIR / "youtube" / "conversations.jsonl",
        SYNTHETIC_DIR / "synthetic_conversations.jsonl",
    ]
    for source in sources:
        data = load_jsonl(source)
        print(f"Loaded {len(data)} from {source}")
        all_conversations.extend(data)

    if not all_conversations:
        print("No data found. Run scrapers and synthetic generator first.")
        return

    print(f"Total loaded: {len(all_conversations)}")

    # Clean
    for conv in all_conversations:
        for msg in conv["messages"]:
            msg["content"] = clean_text(msg["content"])

    # Filter and deduplicate
    valid = [c for c in all_conversations if is_valid(c)]
    print(f"After filtering: {len(valid)}")
    unique = deduplicate(valid)
    print(f"After deduplication: {len(unique)}")

    # Split
    split_idx = int(len(unique) * TRAIN_RATIO)
    train = unique[:split_idx]
    val = unique[split_idx:]

    # Save
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for conv in data:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Train: {len(train)} | Val: {len(val)}")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
