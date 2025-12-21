from datasets import load_dataset, Dataset
import re
import os
from collections import deque

base_dir = "/mnt/d/workspace_backup/workspace/ai/nanochat_eng/.cache"
TOKENIZER_DATASET_PATH = os.path.join(base_dir, "tokenizer_dataset")
REDDIT_DATASET_PATH = os.path.join(base_dir, "reddit_dataset")
GUTENBERG_DATASET_PATH = os.path.join(base_dir, "gutenberg_dataset")

class FastTextShardWriter:
    def __init__(self, out_dir, shard_size_chars=100_000_000, file_buffer_size=1 << 20):
        """
        out_dir: directory to write shards
        shard_size_chars: target chars per shard
        file_buffer_size: Python file-buffering size in bytes (helps reduce syscalls)
        """
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.shard_size_chars = shard_size_chars
        self.file_buffer_size = file_buffer_size

        self.shard_idx = 0
        self.current_chars = 0   # chars already in current shard
        self.file = None

        # deque of strings waiting to be written
        self.chunks = deque()
        self.buffer_chars = 0    # total chars in chunks deque

        self._open_new_shard()

    def _open_new_shard(self):
        if self.file:
            self.file.close()
        path = os.path.join(self.out_dir, f"wiki40b_shard_{self.shard_idx:05d}.txt")
        # bigger buffering to reduce syscalls
        self.file = open(path, "w", buffering=self.file_buffer_size)
        self.shard_idx += 1
        self.current_chars = 0

    def write(self, text: str):
        """
        Append text to buffer. Write to disk only when we can fill shards.
        Guarantees no shard exceeds shard_size_chars.
        """
        if not text:
            return

        # append new chunk
        self.chunks.append(text)
        self.buffer_chars += len(text)

        # while we have enough to fill current shard, consume exactly what's needed
        while self.buffer_chars + self.current_chars >= self.shard_size_chars:
            remaining = self.shard_size_chars - self.current_chars
            to_write_parts = []
            written = 0

            # consume chunks until we have 'remaining' characters
            while remaining > 0 and self.chunks:
                chunk = self.chunks.popleft()
                chunk_len = len(chunk)
                if chunk_len <= remaining:
                    to_write_parts.append(chunk)
                    written += chunk_len
                    remaining -= chunk_len
                    self.buffer_chars -= chunk_len
                else:
                    # partial consume
                    to_write_parts.append(chunk[:remaining])
                    # push leftover back to front
                    leftover = chunk[remaining:]
                    self.chunks.appendleft(leftover)
                    written += remaining
                    self.buffer_chars -= remaining
                    remaining = 0

            # write joined parts (only the exact amount needed)
            self.file.write("".join(to_write_parts))
            self.file.flush()  # optional; can remove for speed, but safer to flush per shard
            self.current_chars += written

            # current shard full -> rotate
            if self.current_chars >= self.shard_size_chars:
                self._open_new_shard()

    def close(self):
        # write any remaining buffered text to the last shard (may be smaller than shard_size_chars)
        if self.buffer_chars > 0:
            # write all remaining chunks
            self.file.write("".join(self.chunks))
            self.buffer_chars = 0
            self.chunks.clear()
        if self.file:
            self.file.close()
            self.file = None


marker_pattern = re.compile(r'\n_START_ARTICLE_\n|\n_START_PARAGRAPH_\n|\n_START_SECTION_\n|\n_NEWLINE_\n')
reddit_junk_pattern = re.compile(r"&gt;//#\w+|&gt;``|&gt;`\*.*?\*`|#\w+")

def download_tokenizer_dataset():

    if os.path.exists(TOKENIZER_DATASET_PATH):
        print("Dataset already present. Skipping download")
        return
    
    print("Downloading tokenizer dataset...")
    os.makedirs(TOKENIZER_DATASET_PATH, exist_ok=True)
    dataset = load_dataset("google/wiki40b", "en", split=None, streaming=True)

    writer = FastTextShardWriter(out_dir=TOKENIZER_DATASET_PATH, shard_size_chars=100_000_000)

    for split in ["test", "validation"]:
        for x in dataset[split]:
            cleaned_text = marker_pattern.sub(" ", x["text"])
            # optional: add newline between examples to avoid bleed
            writer.write(cleaned_text + "\n")

    writer.close()

def download_reddit_comments_dataset():
    reddit_commnents_data = load_dataset("sentence-transformers/reddit", streaming=True)
    writer = FastTextShardWriter(out_dir=REDDIT_DATASET_PATH, shard_size_chars=100_000_000)
    max_chars = 300_000_000
    seen_chars = 0
    for x in reddit_commnents_data["train"]:
        raw_text = x["title"] + " " + x["body"]
        cleaned_text = reddit_junk_pattern.sub("", raw_text)
        writer.write(cleaned_text + "\n")
        seen_chars += len(cleaned_text)
        if seen_chars >= max_chars:
            break
    writer.close()


def download_gutenberg_dataset():
    gutenberg_dataset = load_dataset("Navanjana/Gutenberg_books", streaming=True)
    writer = FastTextShardWriter(out_dir=GUTENBERG_DATASET_PATH, shard_size_chars=100_000_000)
    max_chars = 700_000_000
    seen_chars = 0
    for x in gutenberg_dataset["train"]:
        raw_text = x["paragraph"]
        if not raw_text:
            continue
        writer.write(raw_text + "\n")
        seen_chars += len(raw_text)
        if seen_chars >= max_chars:
            break
    writer.close()

download_tokenizer_dataset()
download_reddit_comments_dataset()
download_gutenberg_dataset()
