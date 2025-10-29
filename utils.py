# utils.py
import re
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[\s.-])?(?:\(?\d{2,4}\)?[\s.-])?\d{3,4}[\s.-]?\d{3,4}")

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_contacts(text: str) -> Dict[str, Optional[str]]:
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    return {
        "emails": list(set(emails)),
        "phones": [p[0] if isinstance(p, tuple) else p for p in phones]
    }

def normalize_text(s: str) -> str:
    return " ".join(s.split())

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def timestamp():
    """Return a filename-safe UTC timestamp.

    Windows disallows characters like ':' in filenames; use a compact
    ISO-like format without forbidden characters.
    Example: 20251028T134551
    """
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")

def dataframe_save(df, path):
    ensure_dir(path)
    df.to_parquet(path, index=False)
