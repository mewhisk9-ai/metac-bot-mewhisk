"""Persist per-model forecasts and maintain accuracy-based ensemble weights."""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("mewhisk.model_memory")

DEFAULT_PATH = Path(os.getenv("MODEL_MEMORY_PATH", "data/model_forecasts.jsonl"))
WEIGHTS_PATH = Path(os.getenv("MODEL_WEIGHTS_PATH", "data/model_weights.json"))


class ModelMemoryStore:
    """
    Append-only JSONL of per-model forecasts, plus derived weights.

    Weight update (when resolutions are recorded):
      weight_i ∝ 1 / (eps + mean_brier_i)
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_PATH,
        weights_path: Path | str = WEIGHTS_PATH,
    ) -> None:
        self.path = Path(path)
        self.weights_path = Path(weights_path)
        self.weights: dict[str, float] = {}
        self._load_weights()

    def _load_weights(self) -> None:
        if not self.weights_path.exists():
            return
        try:
            self.weights = {
                k: float(v) for k, v in json.loads(self.weights_path.read_text()).items()
            }
        except Exception as exc:
            logger.warning("Failed to load model weights: %s", exc)

    def save_weights(self) -> None:
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights_path.write_text(json.dumps(self.weights, indent=2, sort_keys=True))

    def record_forecast(
        self,
        *,
        question_id: Any,
        question_text: str,
        model_id: str,
        question_type: str,
        prediction: Any,
        reasoning: str | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question_id": question_id,
            "question_text": (question_text or "")[:500],
            "model_id": model_id,
            "question_type": question_type,
            "prediction": prediction,
            "reasoning": (reasoning or "")[:4000],
            "resolved": False,
            "outcome": None,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def iter_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def mark_resolution(self, question_id: Any, outcome: float | bool) -> None:
        """Rewrite file marking matching binary rows resolved; then refit weights."""
        y = 1.0 if bool(outcome) else 0.0
        rows = self.iter_rows()
        changed = False
        for row in rows:
            if row.get("question_id") == question_id and row.get("question_type") == "binary":
                row["resolved"] = True
                row["outcome"] = y
                changed = True
        if not changed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        self.refit_weights()

    def refit_weights(self) -> None:
        briers: dict[str, list[float]] = {}
        for row in self.iter_rows():
            if not row.get("resolved") or row.get("question_type") != "binary":
                continue
            model = row.get("model_id")
            pred = row.get("prediction")
            y = row.get("outcome")
            if model is None or pred is None or y is None:
                continue
            try:
                p = float(pred)
                briers.setdefault(str(model), []).append((p - float(y)) ** 2)
            except (TypeError, ValueError):
                continue
        if not briers:
            return
        raw = {
            m: 1.0 / (1e-3 + (sum(vs) / len(vs))) for m, vs in briers.items() if vs
        }
        total = sum(raw.values()) or 1.0
        self.weights = {m: w / total for m, w in raw.items()}
        self.save_weights()
        logger.info("Updated model weights: %s", self.weights)

    def weight_for(self, model_id: str, default: float = 1.0) -> float:
        if not self.weights:
            return default
        return float(self.weights.get(model_id, default * 0.5))

    def weighted_median_binary(
        self, model_ids: list[str], forecasts: list[float]
    ) -> float:
        """Weighted median; falls back to ordinary median if weights missing."""
        if not forecasts:
            raise ValueError("no forecasts")
        if len(forecasts) == 1:
            return float(forecasts[0])
        pairs = sorted(
            zip(forecasts, [self.weight_for(m) for m in model_ids]),
            key=lambda x: x[0],
        )
        total_w = sum(w for _, w in pairs) or 1.0
        acc = 0.0
        for p, w in pairs:
            acc += w
            if acc >= total_w / 2:
                return float(p)
        return float(pairs[-1][0])

    def weighted_mean_vector(
        self, model_ids: list[str], vectors: list[list[float]]
    ) -> list[float]:
        if not vectors:
            raise ValueError("no vectors")
        weights = [self.weight_for(m) for m in model_ids]
        wsum = sum(weights) or 1.0
        dim = len(vectors[0])
        out = []
        for j in range(dim):
            out.append(sum(weights[i] * vectors[i][j] for i in range(len(vectors))) / wsum)
        # renormalize
        s = sum(out) or 1.0
        return [max(1e-6, x) / s for x in out]


def serialize_prediction(pred: Any) -> Any:
    """Convert forecast objects into JSON-serializable values."""
    if isinstance(pred, (int, float, str, bool)) or pred is None:
        return pred
    if hasattr(pred, "predicted_options"):
        return {
            getattr(o, "option_name", str(o)): float(getattr(o, "probability", 0.0))
            for o in pred.predicted_options
        }
    if hasattr(pred, "declared_percentiles"):
        return [
            {"percentile": float(p.percentile), "value": float(p.value)}
            for p in pred.declared_percentiles
        ]
    return str(pred)
