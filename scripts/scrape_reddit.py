"""Scrape r/thaithai and r/Thailand comments as conversation pairs."""
import json
import os
from pathlib import Path

import praw
from praw.models import MoreComments

SUBREDDITS = ["thaithai", "Thailand"]
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped" / "reddit"
LIMIT = 500  # posts per subreddit


def main():
    try:
        _main()
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        print("Required: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET")
        exit(1)
    except Exception as e:
        print(f"Scrape failed: {e}")
        exit(1)


def _main():
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="NeuroThai/1.0",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "conversations.jsonl"
    total = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for sub in SUBREDDITS:
            print(f"Scraping r/{sub}")
            for post in reddit.subreddit(sub).hot(limit=LIMIT):
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    if isinstance(comment, MoreComments):
                        continue
                    if len(comment.body) < 20:
                        continue

                    # Post body -> comment as user->assistant
                    parent_text = post.selftext or post.title
                    if len(parent_text) >= 15:
                        pair = {
                            "messages": [
                                {"role": "user", "content": parent_text[:2048]},
                                {"role": "assistant", "content": comment.body[:2048]},
                            ]
                        }
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        f.flush()
                        total += 1

                    # Comment chain: parent comment -> reply
                    try:
                        parent = comment.parent()
                    except Exception:
                        parent = None
                    if isinstance(parent, praw.models.Comment) and len(parent.body) >= 15:
                        pair = {
                            "messages": [
                                {"role": "user", "content": parent.body[:2048]},
                                {"role": "assistant", "content": comment.body[:2048]},
                            ]
                        }
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        f.flush()
                        total += 1

    print(f"Saved {total} conversation pairs to {out_path}")


if __name__ == "__main__":
    main()
