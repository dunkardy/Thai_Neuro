# Neuro-Thai Persona Model

## Interim progress rule
When running long operations (data generation, training, scraping, evaluation), always show per-item interim progress so the user can verify correctness in real time — not just per-batch or per-stage summaries. Print a short preview of each result as it completes (e.g., first line of a generated conversation, snippet of a scraped post). Stream-write output files incrementally with flush so partial results survive interruptions.

## Project structure
- `training/` — data prep, synthetic generation, QLoRA training, evaluation
- `inference/` — CLI chat interface
- `scripts/` — web scrapers (Pantip, Reddit, YouTube)
- `data/` — raw, processed, synthetic, eval datasets
- `adapters/` — saved LoRA weights

## Key technical constraints
- GTX 1660 Ti Max-Q (6 GB VRAM) — always use float16, NOT bfloat16
- DeepSeek API via OpenAI SDK — base_url="https://api.deepseek.com"
- Training uses Unsloth QLoRA with Gemma 4 E4B abliterated
- Thai language: colloquial Gen Z style with natural code-switching
- All config in `training/config.py` — single source of truth
