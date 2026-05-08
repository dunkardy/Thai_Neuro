"""Scrape YouTube comments from popular Thai channels."""
import json
import os
import time
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

THAI_CHANNEL_IDS = []
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


def get_comments(youtube, video_id: str) -> list[str]:
    """Get top-level comments from a video."""
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=MAX_COMMENTS,
            order="relevance",
        )
        response = execute_with_retry(request, label=f"comments for video {video_id}")
        if response is None:
            return comments
        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            if len(text) >= 15:
                comments.append(text)
    except Exception as e:
        print(f"  Error getting comments for video {video_id}: {e}")
    return comments


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
            print(f"Scraping channel: {channel_id}")
            video_ids = get_video_ids(youtube, channel_id)
            print(f"  Found {len(video_ids)} videos")
            for vid in video_ids:
                comments = get_comments(youtube, vid)
                for c in comments:
                    pair = {
                        "messages": [
                            {"role": "user", "content": "พูดถึงเรื่องนี้หน่อย"},
                            {"role": "assistant", "content": c[:2048]},
                        ]
                    }
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    f.flush()
                    total += 1
                time.sleep(REQUEST_DELAY)

    print(f"Saved {total} comments to {out_path}")


if __name__ == "__main__":
    main()
