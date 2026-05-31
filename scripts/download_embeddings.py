"""Download Chinese Word2Vec embeddings for DADGNN.

Primary: Tencent AI Lab Chinese Embedding (200d, 143k words, 64% coverage)
Fallback: pangda/word2vec-skip-gram-mixed-large-chinese (300d, 18k words)
"""

import os
import sys
import urllib.request

EMBEDDINGS = {
    "Tencent_AILab_ChineseEmbedding.bin": {
        "url": "https://huggingface.co/shibing624/text2vec-word2vec-tencent-chinese/resolve/main/light_Tencent_AILab_ChineseEmbedding.bin",
        "desc": "Tencent AI Lab Chinese Embedding (200d, 143k words)",
    },
    "sgns.merge.word": {
        "url": "https://huggingface.co/pangda/word2vec-skip-gram-mixed-large-chinese/resolve/main/embedding.txt",
        "desc": "Chinese Word2Vec SGNS (300d, 18k words)",
    },
}


def download_file(url: str, out_path: str, desc: str):
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Already exists: {out_path} ({size_mb:.1f} MB)")
        return out_path

    print(f"  Downloading {desc}...")
    print(f"  URL: {url}")
    print(f"  Output: {out_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        mb = downloaded / 1024 / 1024
        sys.stdout.write(f"\r  Progress: {mb:.1f} MB ({pct:.0f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, out_path, progress_hook)
    print(f"\n  Done! ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)")
    return out_path


def download_embeddings(data_dir: str | None = None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "homework2", "data")

    os.makedirs(data_dir, exist_ok=True)

    for filename, info in EMBEDDINGS.items():
        out_path = os.path.join(data_dir, filename)
        download_file(info["url"], out_path, info["desc"])

    print("\nAll embeddings downloaded.")


if __name__ == "__main__":
    download_embeddings()
