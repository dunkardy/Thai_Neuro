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
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent="NeuroThai/1.0",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = []

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
                    all_pairs.append({
                        "messages": [
                            {"role": "user", "content": parent_text[:2048]},
                            {"role": "assistant", "content": comment.body[:2048]},
                        ]
                    })

                # Comment chain: parent comment -> reply
                parent = comment.parent()
                if isinstance(parent, praw.models.Comment) and len(parent.body) >= 15:
                    all_pairs.append({
                        "messages": [
                            {"role": "user", "content": parent.body[:2048]},
                            {"role": "assistant", "content": comment.body[:2048]},
                        ]
                    })

    out_path = OUTPUT_DIR / "conversations.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_pairs)} conversation pairs to {out_path}")


if __name__ == "__main__":
    main()
