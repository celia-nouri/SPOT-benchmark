import sqlite3
import csv
import sys


DB_PATH = "data/fr_subreddits_2026-02.sqlite"   # ← adjust path
OUTPUT_PATH = "data/fr_armin_reddit_comments.csv"
BATCH_SIZE = 5000

SQL = """
SELECT
    c.reddit_id         AS comment_id,
    c.permalink,
    c.author,
    c.subreddit,
    c.body              AS text,
    c.score,
    c.created_utc,
    c.parent_id,
    c.link_id           AS post_id,
    s.title             AS post_title,
    s.selftext          AS post_message,
    s.domain            AS parent_domain,
    s.permalink         AS post_permalink,
    pc.body             AS parent_text
FROM comment c
LEFT JOIN submission s
    ON SUBSTR(c.link_id, 4) = s.reddit_id
LEFT JOIN comment pc
    ON c.parent_id = pc.reddit_id
"""

COLUMNS = [
    "comment_id", "permalink", "account_name", "subreddit",
    "text", "score", "created_utc", "parent_id", "post_id",
    "post_title", "post_description", "post_message",
    "parent_domain", "post_permalink", "parent_text",
]

def export(db_path: str, output_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size=-131072;")   # 128 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY;")

    cur = conn.cursor()
    cur.execute(SQL)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(COLUMNS)

        total = 0
        while True:
            rows = cur.fetchmany(BATCH_SIZE)
            if not rows:
                break
            writer.writerows(rows)
            total += len(rows)
            print(f"\r  {total:,} rows written...", end="", flush=True)

    conn.close()
    print(f"\nDone → {output_path}  ({total:,} rows)")

if __name__ == "__main__":
    export(DB_PATH, OUTPUT_PATH)

