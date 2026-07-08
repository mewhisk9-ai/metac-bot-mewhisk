"""Lightweight RAG over past bot rationales for similar questions."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("mewhisk.rationale_rag")

DEFAULT_PATH = Path(os.getenv("RATIONALE_RAG_PATH", "data/rationales.jsonl"))

_STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "for",
    "and",
    "or",
    "will",
    "be",
    "by",
    "at",
    "is",
    "are",
    "was",
    "were",
    "with",
    "from",
    "that",
    "this",
    "it",
    "as",
    "before",
    "after",
    "than",
    "more",
    "less",
    "how",
    "what",
    "when",
    "which",
    "who",
    "probability",
    "forecast",
    "question",
}


def _tokenize(text: str) -> list[str]:
    return [
        w
        for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in _STOP
    ]


class RationaleStore:
    """
    Stores past question→rationale pairs and retrieves similar ones via TF-IDF cosine.
    No external embedding API required.
    """

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._docs: list[dict[str, Any]] = []
        self._dfs: Counter[str] = Counter()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        docs = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        self._docs = docs
        self._recompute_df()
        logger.info("Loaded %d past rationales for RAG", len(self._docs))

    def _recompute_df(self) -> None:
        self._dfs = Counter()
        for doc in self._docs:
            toks = set(_tokenize(doc.get("question_text", "")))
            self._dfs.update(toks)

    def _tfidf(self, text: str) -> dict[str, float]:
        toks = _tokenize(text)
        if not toks:
            return {}
        tf = Counter(toks)
        n = max(len(self._docs), 1)
        vec = {}
        for term, count in tf.items():
            df = self._dfs.get(term, 0) + 1  # smooth
            idf = math.log((n + 1) / df) + 1.0
            vec[term] = (count / len(toks)) * idf
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        keys = set(a) & set(b)
        num = sum(a[k] * b[k] for k in keys)
        da = math.sqrt(sum(v * v for v in a.values()))
        db = math.sqrt(sum(v * v for v in b.values()))
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)

    def add(
        self,
        *,
        question_id: Any,
        question_text: str,
        prediction: Any,
        reasoning: str,
        question_type: str = "unknown",
    ) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question_id": question_id,
            "question_text": question_text,
            "question_type": question_type,
            "prediction": prediction,
            "reasoning": (reasoning or "")[:6000],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self._docs.append(row)
        # Incremental DF update
        self._dfs.update(set(_tokenize(question_text)))

    def similar(
        self, question_text: str, *, k: int = 3, min_score: float = 0.12
    ) -> list[tuple[float, dict[str, Any]]]:
        if not self._docs:
            return []
        qv = self._tfidf(question_text)
        scored = []
        for doc in self._docs:
            # Avoid exact same question text dominating
            if doc.get("question_text") == question_text:
                continue
            s = self._cosine(qv, self._tfidf(doc.get("question_text", "")))
            if s >= min_score:
                scored.append((s, doc))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def format_context(self, question_text: str, *, k: int = 3) -> str:
        hits = self.similar(question_text, k=k)
        if not hits:
            return ""
        lines = [
            "--- SIMILAR PAST RATIONALES ---",
            "Past bot rationales on related questions (for analogy only; "
            "do not copy probabilities blindly):",
        ]
        for score, doc in hits:
            pred = doc.get("prediction")
            reason = (doc.get("reasoning") or "").strip().replace("\n", " ")
            if len(reason) > 700:
                reason = reason[:700] + "…"
            lines.append(
                f"- (sim={score:.2f}) Q: {doc.get('question_text')}\n"
                f"  Prior forecast: {pred}\n"
                f"  Rationale excerpt: {reason}"
            )
        return "\n".join(lines) + "\n"
