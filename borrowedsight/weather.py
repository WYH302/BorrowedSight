"""Train jitter versus disjoint LISA test weather.

Training uses light point drop and jitter only.
Test fog is two-way Beer-Lambert. Test rain adds attenuation plus early
backscatter returns. Those coefficients are not used in training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from borrowedsight.paths import CONFIGS


def load_weather(path: Path | None = None) -> dict:
    path = path or (CONFIGS / "weather.yaml")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert_disjoint(cfg)
    return cfg


def assert_disjoint(cfg: dict) -> None:
    train = cfg["train"]
    fog = cfg["test_fog"]
    rain = cfg["test_rain"]
    banned = {"alpha_per_m", "rainfall_mm_h", "backscatter_prob_per_m"}
    if set(train) & banned:
        raise AssertionError("training augmentation must not carry LISA coefficients")
    if {"drop_prob", "jitter_std_m"} & set(fog):
        raise AssertionError("fog test must not reuse train drop/jitter")
    if {"drop_prob", "jitter_std_m"} & set(rain):
        raise AssertionError("rain test must not reuse train drop/jitter")
    if not fog.get("two_way") or not rain.get("two_way"):
        raise AssertionError("test weather uses two-way Beer-Lambert")


def _ranges(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points, axis=1)


def train_degrade(points: np.ndarray, rng: np.random.Generator, cfg: dict | None = None) -> np.ndarray:
    cfg = (cfg or load_weather())["train"]
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.copy()
    keep = rng.random(pts.shape[0]) >= float(cfg["drop_prob"])
    pts = pts[keep]
    if pts.size == 0:
        return pts
    return pts + rng.normal(0.0, float(cfg["jitter_std_m"]), size=pts.shape)


def lisa_fog(points: np.ndarray, rng: np.random.Generator, cfg: dict | None = None) -> np.ndarray:
    """Keep each return with probability exp(-2 * alpha * range)."""
    cfg = (cfg or load_weather())["test_fog"]
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.copy()
    keep_p = np.exp(-2.0 * float(cfg["alpha_per_m"]) * _ranges(pts))
    return pts[rng.random(pts.shape[0]) < keep_p]


def lisa_rain(
    points: np.ndarray, rng: np.random.Generator, cfg: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Attenuate and insert closer backscatter returns.

    The second array is 1 on backscatter rows and 0 on surviving original rows.
    """
    cfg = (cfg or load_weather())["test_rain"]
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.copy(), np.zeros((0,), dtype=np.uint8)
    ranges = _ranges(pts)
    keep_p = np.exp(-2.0 * float(cfg["alpha_per_m"]) * ranges)
    keep = rng.random(pts.shape[0]) < keep_p
    kept = pts[keep]
    kept_range = ranges[keep]
    if kept.size == 0:
        return kept, np.zeros((0,), dtype=np.uint8)
    gamma = float(cfg["backscatter_prob_per_m"])
    trigger = rng.random(kept.shape[0]) < (1.0 - np.exp(-gamma * kept_range))
    flags = np.zeros(kept.shape[0], dtype=np.uint8)
    if not np.any(trigger):
        return kept, flags
    frac = np.clip(rng.random(int(trigger.sum())), 0.05, 0.95)
    extra = kept[trigger] * frac[:, None]
    extra_flags = np.ones(extra.shape[0], dtype=np.uint8)
    return np.concatenate([kept, extra], axis=0), np.concatenate([flags, extra_flags], axis=0)
