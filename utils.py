# utils.py
import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    # Basic sentence-aware splitter
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        if end >= L:
            chunks.append(text[start:L])
            break
        # try to break at sentence boundary within last 200 chars
        window = text[start:end]
        last_period = window.rfind('. ')
        if last_period != -1 and last_period > chunk_size - 300:
            split_at = start + last_period + 1
        else:
            split_at = end
            chunks.append(text[start:split_at].strip())
            start = max(0, split_at - overlap)
    return [c for c in chunks if c]