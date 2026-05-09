"""Scrape YouTube comments from popular Thai channels."""
import json
import os
import time
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

THAI_CHANNEL_IDS = [
    # Gaming
    "UCAcYshki68X7GXNrYnbZQEQ",  # Bon Gamerz
    "UCckBCIvFC0zKOtyz3F7igRQ",  # Basgamer
    "UCU0ou7IwjcwqZoChOHN85pQ",  # PIGCEL
    "UCt-CLFyfQjadm6w7ZgDVjaQ",  # พี่หมีฟูจิ
    "UCiHJOhRg-1bxHh7t_PLppNw",  # sonjokey
    "UCpQAzHYc2q_rw2banp5uUzw",  # Moment Games
    "UC705OZcYT8ysriJ93eiQVeg",  # NottO NBK
    "UC4ytWAm10LjaeS53LpQfJOA",  # Kutcha Wants2playz
    "UC562Vvs8XM-0FJ0C98c5Rnw",  # MORLHAM GAMER
    "UCaPAgnV8Bh0upyayerSRGJg",  # The Xesitz
    "UCbFBdunU9rTHIooyxLmi1MQ",  # PARIWAT
    "UClns-mkWsQ4V7fNtZ-U-36w",  # HookHuukGaming
    "UCe5H8YhDjg_441KjjS5qrAg",  # Garena Free Fire TH
    "UCPryqo3M_mit4_A5gD99LtQ",  # Phillthesun
    # Vlog / Lifestyle
    "UC9NV3GWEacZDSSKK3oRtjcw",  # Fizart VLOG
    "UCUHf9-Q9AtN_l4IWpuYzP4A",  # Butories
    "UCghqIDNjD2B2CvgALoLxjkw",  # gap.bumseeker
    "UCAlfop5P8slp0VewzUjj8GA",  # Jayy Crane
    "UC2I1Ye8xVHLC-iQJXMLT-Gg",  # DEEN VLOG
    "UComAntq7Iq5SxYHUJuJ6ekg",  # BACKPAEGER
    "UCRo1j9THNtcvVa4vi5xCn0g",  # โลกของคนมีหนวด
    "UCoJuMpfYKqCbvnd1DpUUJmg",  # พิมรี่พาย
    # Entertainment
    "UCWdWbuzG0YzPoo9rcWEW2Vg",  # HaGate Studio
    "UCz5BjXkFFTuBsMx5MSRYwkA",  # Gapo Entertainment
    # Music
    "UCv9Xh8IDubBtFvoYfgCUbRQ",  # KRK Music
    "UCoMkFgTybLK4UJYA_nLp1ZA",  # SONG RIDER
    "UCtqBOzg93aIu8V-VgjRdWnA",  # SERNG MUSiC
    "UC-vShcsYE730gWSbvsy44QQ",  # RISER MUSIC
    "UCm-Cz0JEMr0_EIyWE7lvRdg",  # Thaidol Music
    "UCQ9v04cmBdKAUbR4-SH6v2g",  # YOUNGOHM
    "UCffA3uduAMx9YArnog42p2A",  # BLE PATUMRACH
    "UCQHd_FYpAY4MJy7to9w4bKA",  # DIAMOND MQT
    # Review / Tech
    "UCbLxbHXO0pkWamgcJ6N9x1Q",  # Djung TV
    "UCZ1xUPnSDPRtz76nGNBcaIA",  # Techcast
    # Funny
    "UCQtOWW-F-LQwrMh0hhBur5g",  # Funny Thailand
    "UCjvmqQQIntIY-x_KPyHdwag",  # Tawan Funny
    "UC7im0hCrsv7qODm75ZiazNw",  # FunnY เมืองไทย
]
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped" / "youtube"
MAX_COMMENTS = 100
MAX_VIDEOS = 50
MAX_RETRIES = 3
REQUEST_DELAY = 1.0  # seconds between API calls


def execute_with_retry(request, label: str = ""):
    """Execute a YouTube API request with retry and backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return request.execute()
        except HttpError as e:
            status = e.resp.status if hasattr(e, "resp") else None
            if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) * 2
                print(f"  API error {status}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"  HttpError for {label}: {e}")
            return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) * 2
                print(f"  Request failed: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Request failed after {MAX_RETRIES} attempts for {label}: {e}")
                return None
    return None


def get_video_ids(youtube, channel_id: str, max_videos: int = MAX_VIDEOS) -> list[str]:
    """Get recent video IDs from a channel."""
    video_ids = []
    uploads_playlist = "UU" + channel_id[2:]
    try:
        request = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist,
            maxResults=max_videos,
        )
        while request:
            response = execute_with_retry(
                request, label=f"videos for channel {channel_id}"
            )
            if response is None:
                break
            for item in response.get("items", []):
                video_ids.append(item["snippet"]["resourceId"]["videoId"])
            request = youtube.playlistItems().list_next(request, response)
            time.sleep(REQUEST_DELAY)
    except Exception as e:
        print(f"  Error getting videos for channel {channel_id}: {e}")
    return video_ids


def _clean_comment(text: str) -> str:
    """Strip HTML tags/entities, @mentions, normalize whitespace."""
    import html
    import re
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"\+\S+", "", text)
    text = " ".join(text.split())
    return text if len(text) >= 15 else ""


def get_comment_chains(youtube, video_id: str) -> list[dict]:
    """Extract reply chains as multi-turn conversations.

    Returns list of {"messages": [{"role": ..., "content": ...}, ...]} dicts.
    Each chain starts with the top comment as user, then alternates
    assistant/user through the reply thread.
    """
    chains = []
    try:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=MAX_COMMENTS,
            order="relevance",
        )
        response = execute_with_retry(request, label=f"comments for video {video_id}")
        if response is None:
            return chains

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            top_text = _clean_comment(top["textDisplay"])
            if not top_text:
                continue

            replies = item.get("replies", {}).get("comments", [])
            if not replies:
                # Single top comment, no replies — skip (can't make a pair)
                continue

            # Build multi-turn chain: top comment (user) -> reply1 (assistant) -> ...
            messages = [{"role": "user", "content": top_text[:2048]}]
            for i, reply in enumerate(replies):
                reply_text = _clean_comment(reply["snippet"]["textDisplay"])
                if not reply_text:
                    continue
                role = "assistant" if i % 2 == 0 else "user"
                messages.append({"role": role, "content": reply_text[:2048]})

            if len(messages) >= 2:  # at least one user + one assistant
                chains.append({"messages": messages})

    except Exception as e:
        print(f"  Error getting comments for video {video_id}: {e}")
    return chains


def main():
    try:
        _main()
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        print("Required: YOUTUBE_API_KEY")
        exit(1)
    except Exception as e:
        print(f"Scrape failed: {e}")
        exit(1)


def _main():
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("YOUTUBE_API_KEY environment variable not set")
        return

    if not THAI_CHANNEL_IDS:
        print("THAI_CHANNEL_IDS is empty — edit the script to add Thai YouTube channel IDs")
        return

    youtube = build("youtube", "v3", developerKey=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUTPUT_DIR / "conversations.jsonl"
    total = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for channel_id in THAI_CHANNEL_IDS:
            print(f"\nScraping channel: {channel_id}")
            video_ids = get_video_ids(youtube, channel_id)
            print(f"  Found {len(video_ids)} videos")
            for vi, vid in enumerate(video_ids):
                chains = get_comment_chains(youtube, vid)
                for chain in chains:
                    f.write(json.dumps(chain, ensure_ascii=False) + "\n")
                    f.flush()
                    total += 1
                if chains:
                    sample = chains[0]
                    n_turns = len(sample["messages"])
                    first = sample["messages"][0]["content"][:50]
                    preview = first + ("..." if len(sample["messages"][0]["content"]) > 50 else "")
                    print(f"  [{vi+1}/{len(video_ids)}] {len(chains)} chains ({n_turns}t) — {preview}")
                time.sleep(REQUEST_DELAY)

    print(f"Saved {total} comments to {out_path}")


if __name__ == "__main__":
    main()
