from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)


def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding=encoding)


def read_json(path: Path, encoding: str = "utf-8") -> Any:
    return json.loads(read_text(path, encoding=encoding))


def write_json(path: Path, obj: Any, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding=encoding)


def iter_jsonl(path: Path, encoding: str = "utf-8") -> Iterable[Any]:
    with path.open("r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, items: Iterable[Any], encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding=encoding) as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


_whitespace_re = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _whitespace_re.sub(" ", text).strip()


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    keep = (max_chars - 3) // 2
    return f"{text[:keep]}...{text[-keep:]}"


def simple_tokenize(text: str) -> set[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+|[\u4e00-\u9fff]+", text)
    return {t for t in tokens if t and len(t) >= 2}


def jaccard_similarity(a: str, b: str) -> float:
    sa = simple_tokenize(a)
    sb = simple_tokenize(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0
