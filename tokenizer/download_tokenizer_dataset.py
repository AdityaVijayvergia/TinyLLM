from datasets import load_dataset, Dataset
import re
import os

base_dir = "/mnt/d/workspace_backup/workspace/ai/nanochat_eng/.cache"
TOKENIZER_DATASET_PATH = os.path.join(base_dir, "tokenizer_dataset")

class TextShardWriter:
    def __init__(self, shard_size_chars=100_000_000):
        """
        shard_size_chars:
            Approximate number of characters per shard
            (100M chars ≈ ~100MB)
        """
        
        self.shard_size_chars = shard_size_chars
        self.shard_idx = 0
        self.file = None
        self.buffer = ""

        self._open_new_shard()

    def _open_new_shard(self):
        if self.file:
            self.file.close()

        path = os.path.join(
            TOKENIZER_DATASET_PATH,
            f"wiki40b_shard_{self.shard_idx:05d}.txt"
        )
        self.file = open(path, "w")
        self.shard_idx += 1

    def write(self, text: str):
        if not text:
            return

        text = self.buffer + text
        self.buffer = ""
        text_len = len(text)
        start = 0
        # print(f"Text length: {text_len}, shard: {self.shard_idx}")
        while text_len > self.shard_size_chars:
            print(f"Writing shard {self.shard_idx}")
            self.file.write(text[start:start + self.shard_size_chars])
            self._open_new_shard()
            start += self.shard_size_chars
            text_len -= self.shard_size_chars
        
        self.buffer = text[start:]

    def close(self):
        if self.file:
            self.file.write(self.buffer)
            self.file.close()
            print("Closing file writer")
            self.file = None


marker_pattern = re.compile(r'\n_START_ARTICLE_\n|\n_START_PARAGRAPH_\n|\n_START_SECTION_\n|\n_NEWLINE_\n')

def download_tokenizer_dataset():

    if os.path.exists(TOKENIZER_DATASET_PATH):
        print("Dataset already present. Skipping download")
        return
    
    print("Downloading tokenizer dataset...")
    os.makedirs(TOKENIZER_DATASET_PATH, exist_ok=True)
    dataset = load_dataset("google/wiki40b", "en", split=None, streaming=True)

    writer = TextShardWriter(shard_size_chars=100_000_000)

    try:
        last = 0
        for split in ["test", "validation"]:
            print(f"============split: {split}")
            for x in dataset[split]:
                cleaned_text = marker_pattern.sub(" ", x["text"])
                writer.write(cleaned_text)
                last+=1
                if last % 1000 == 0:
                    print(last)

    finally:
        writer.close()


download_tokenizer_dataset()
