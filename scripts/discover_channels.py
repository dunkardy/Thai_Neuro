"""Discover popular Thai YouTube channel IDs via the YouTube API."""
import json
import os
import sys
from pathlib import Path

from googleapiclient.discovery import build

KEYWORDS = [
    "คนไทย gaming",
    "คนไทย vlog",
    "คนไทย entertainment ตลก",
    "คนไทย review",
    "คนไทย music เพลง",
    "คนไทย lifestyle แชท",
    "ไทย gaming live",
    "ไทย funny",
]


def main():
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("YOUTUBE_API_KEY not set")
        sys.exit(1)

    youtube = build("youtube", "v3", developerKey=api_key)
    found: dict[str, str] = {}  # channel_id -> title

    for keyword in KEYWORDS:
        print(f"\nSearching: {keyword}")
        try:
            resp = youtube.search().list(
                part="snippet",
                q=keyword,
                type="channel",
                maxResults=10,
                relevanceLanguage="th",
                regionCode="TH",
            ).execute()
        except Exception as e:
            print(f"  Error: {e}")
            continue

        for item in resp.get("items", []):
            cid = item["snippet"]["channelId"]
            title = item["snippet"]["channelTitle"]
            if cid not in found:
                found[cid] = title
                print(f"  {cid}  # {title}")

    print(f"\n--- {len(found)} unique channels found ---")
    print("Copy these into scrape_youtube.py's THAI_CHANNEL_IDS:")
    print(json.dumps(list(found.keys()), indent=4))


if __name__ == "__main__":
    main()
