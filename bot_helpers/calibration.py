"""Historical calibration / extremization for binary forecasts."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("mewhisk.calibration")

DEFAULT_PATH = Path(os.getenv("CALIBRATION_PATH", "data/calibration.json"))


def _clip01(p: float) -> float:
    return float(np.clip(p, 0.01, 0.99))


def extremize(p: float, strength: float = 1.35) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    odds = (p / (1 - p)) ** strength
    return _clip01(odds / (1 + odds))


class CalibrationStore:
    """
    Stores resolved binary outcomes vs predicted probabilities and fits a
    simple logistic calibration + extremization strength.

    File format:
    {
      "records": [{"p": 0.3, "y": 0}, ...],
      "strength": 1.35,
      "a": 0.0,
      "b": 1.0
    }
    where calibrated_p = sigmoid(a + b * logit(p)), then extremize(strength).
    """

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.records: list[dict[str, float]] = []
        self.strength = float(os.getenv("CALIBRATION_STRENGTH", "1.35"))
        self.a = 0.0
        self.b = 1.0
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.records = list(data.get("records", []))
            self.strength = float(data.get("strength", self.strength))
            self.a = float(data.get("a", 0.0))
            self.b = float(data.get("b", 1.0))
            logger.info(
                "Loaded calibration (%d records, strength=%.2f, a=%.3f, b=%.3f)",
                len(self.records),
                self.strength,
                self.a,
                self.b,
            )
        except Exception as exc:
            logger.warning("Failed to load calibration: %s", exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": self.records[-5000:],
            "strength": self.strength,
            "a": self.a,
            "b": self.b,
        }
        self.path.write_text(json.dumps(payload, indent=2))

    def add_resolution(self, predicted: float, outcome: float | bool) -> None:
        y = 1.0 if bool(outcome) else 0.0
        self.records.append({"p": _clip01(float(predicted)), "y": y})
        if len(self.records) >= 20 and len(self.records) % 5 == 0:
            self.refit()
        self.save()

    def refit(self) -> None:
        """Fit logistic a,b and choose extremization strength on holdout Brier."""
        if len(self.records) < 20:
            return
        ps = np.array([r["p"] for r in self.records], dtype=float)
        ys = np.array([r["y"] for r in self.records], dtype=float)
        logits = np.log(ps / (1 - ps))

        # Closed-form-ish: regress y on logit via simple gradient steps
        a, b = 0.0, 1.0
        lr = 0.05
        for _ in range(200):
            z = a + b * logits
            pred = 1 / (1 + np.exp(-z))
            err = pred - ys
            a -= lr * err.mean()
            b -= lr * (err * logits).mean()
        self.a, self.b = float(a), float(b)

        # Pick extremization strength that minimizes Brier after logistic map
        best_s, best_brier = self.strength, 1.0
        for s in np.linspace(1.0, 2.0, 11):
            cal = 1 / (1 + np.exp(-(a + b * logits)))
            cal = np.clip(cal, 1e-6, 1 - 1e-6)
            odds = (cal / (1 - cal)) ** s
            out = odds / (1 + odds)
            brier = float(np.mean((out - ys) ** 2))
            if brier < best_brier:
                best_brier, best_s = brier, float(s)
        self.strength = best_s
        logger.info(
            "Refit calibration: a=%.3f b=%.3f strength=%.2f brier=%.4f",
            self.a,
            self.b,
            self.strength,
            best_brier,
        )
        self.save()

    def apply(self, p: float) -> float:
        p = _clip01(p)
        logit = math.log(p / (1 - p))
        z = self.a + self.b * logit
        cal = 1 / (1 + math.exp(-z))
        return extremize(cal, self.strength)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.records),
            "strength": self.strength,
            "a": self.a,
            "b": self.b,
        }


def apply_binary_calibration(p: float, store: CalibrationStore | None = None) -> float:
    if store is None:
        return extremize(p)
    return store.apply(p)
