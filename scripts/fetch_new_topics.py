#!/usr/bin/env python3
"""
Fetch new DGX Spark forum topics from both the main and projects forums,
compare against already-processed topic IDs, and emit a summary of new
threads for the agent to ingest.

Reads:  sources/processed_topics.txt  (one topic ID per line)
Writes: /tmp/sparkbase_new_topics.json (list of new topic dicts with id/title/url/excerpt/posts_count/views/created_at)

Discourse JSON API: https://forums.developer.nvidia.com/c/<category>/<id>.json
"""

import urllib.request
import json
import os
import sys
import re

CATEGORIES = {
    "main": 721,
    "projects": 723,
}

BASE = "https://forums.developer.nvidia.com"
PROCESSED_FILE = os.path.join(os.path.dirname(__file__), "..", "sources", "processed_topics.txt")
OUTPUT_FILE = "/tmp/sparkbase_new_topics.json"

PAGES_TO_SCAN = 5  # Discourse shows 30 topics/page; 5 pages = 150 recent topics per category


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_processed():
    ids = set()
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE) as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    ids.add(int(line))
    return ids


def scan_category(cat_name, cat_id):
    topics = []
    for page in range(1, PAGES_TO_SCAN + 1):
        url = f"{BASE}/c/accelerated-computing/dgx-spark-gb10/{'dgx-spark-gb10' if cat_name == 'main' else 'dgx-spark-gb10-projects'}/{cat_id}.json?page={page}"
        try:
            data = fetch_json(url)
            page_topics = data.get("topic_list", {}).get("topics", [])
            topics.extend(page_topics)
        except Exception as e:
            print(f"WARN: failed to fetch {cat_name} page {page}: {e}", file=sys.stderr)
            break

    # Deduplicate
    seen = set()
    unique = []
    for t in topics:
        if t["id"] not in seen:
            seen.add(t["id"])
            unique.append(t)
    return unique


def main():
    processed = load_processed()
    all_new = []

    for cat_name, cat_id in CATEGORIES.items():
        topics = scan_category(cat_name, cat_id)
        for t in topics:
            if t["id"] in processed:
                continue
            # Skip pinned/FAQ/welcome/closed topics that are admin posts
            if t.get("pinned"):
                continue
            if t.get("closed"):
                continue

            excerpt = t.get("excerpt", "") or ""
            excerpt = re.sub(r"<[^>]+>", "", excerpt).strip()[:300]

            all_new.append({
                "id": t["id"],
                "title": t.get("title", ""),
                "url": f"{BASE}/t/{t['id']}",
                "excerpt": excerpt,
                "posts_count": t.get("posts_count", 0),
                "views": t.get("views", 0),
                "created_at": t.get("created_at", ""),
                "last_posted_at": t.get("last_posted_at", ""),
                "category": cat_name,
            })

    # Sort by last_posted_at descending (most recent activity first)
    all_new.sort(key=lambda x: x.get("last_posted_at", ""), reverse=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_new, f, indent=2)

    # Emit summary for the agent
    if not all_new:
        print("No new forum topics found. All recent threads are already processed.")
    else:
        print(f"Found {len(all_new)} new forum topic(s) not yet in sparkbase:\n")
        for t in all_new:
            print(f"  [{t['category']}] {t['id']}: {t['title']}")
            print(f"    views={t['views']} posts={t['posts_count']} last={t['last_posted_at'][:10]}")
            print(f"    {t['excerpt'][:150]}")
            print()
        print(f"Full data written to {OUTPUT_FILE}")
        print(f"To ingest: fetch each topic's JSON from {BASE}/t/<id>.json, extract findings,")
        print(f"update wiki pages, register sources, and append topic IDs to {PROCESSED_FILE}")


if __name__ == "__main__":
    main()