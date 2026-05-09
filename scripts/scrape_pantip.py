"""Scrape Pantip threads with comments as conversation pairs."""
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://pantip.com"
TAGS = ["สังคมวัยรุ่น", "ชีวิตวัยรุ่น", "ความรัก", "ปัญหา", "ซุบซิบ"]
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped" / "pantip"
REQUEST_DELAY = 1.5  # seconds between requests
MAX_RETRIES = 3


def fetch(url: str, timeout: int = 30) -> requests.Response | None:
    """Fetch URL with retries and backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url, headers={"User-Agent": "NeuroThai/1.0"}, timeout=timeout
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = (2 ** attempt) * 2
                print(f"  Server error {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) * 2
                print(f"  Request failed: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Request failed after {MAX_RETRIES} attempts: {url} - {e}")
                return None
    return None


def get_topic_urls(tag: str, pages: int = 5) -> list[str]:
    """Collect topic URLs from a tag's listing pages."""
    urls = []
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/tag/{tag}?page={page}"
        resp = fetch(url)
        if resp is None or resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.select("a[href^='/topic/']"):
            href = link.get("href")
            if href:
                urls.append(BASE_URL + href)
        time.sleep(REQUEST_DELAY)
    return urls


def parse_topic(topic_url: str) -> list[dict] | None:
    """Extract topic body and top-level comments as conversation pairs."""
    resp = fetch(topic_url)
    if resp is None or resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    post_body = soup.select_one(".display-post-story")
    if not post_body:
        return None
    post_text = post_body.get_text(strip=True)
    if len(post_text) < 20:
        return None

    # Only process top-level comments to avoid duplicates
    pairs = []
    topic_container = soup.select_one(".post-item")
    if topic_container:
        comments = topic_container.select(":scope > .comment-item")
    else:
        comments = soup.select(".comment-item")

    for comment in comments:
        comment_text = comment.select_one(".display-post-story")
        if not comment_text:
            continue
        reply_text = comment_text.get_text(strip=True)
        if len(reply_text) < 20:
            continue
        pairs.append({
            "messages": [
                {"role": "user", "content": post_text[:2048]},
                {"role": "assistant", "content": reply_text[:2048]},
            ]
        })

    return pairs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    seen = set()

    for tag in TAGS:
        print(f"Scraping tag: {tag}")
        urls = get_topic_urls(tag, pages=5)
        print(f"  Found {len(urls)} topics")
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            pairs = parse_topic(url)
            if pairs:
                all_pairs.extend(pairs)
            time.sleep(REQUEST_DELAY)

    # Save
    out_path = OUTPUT_DIR / "conversations.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_pairs)} conversation pairs to {out_path}")


if __name__ == "__main__":
    main()
