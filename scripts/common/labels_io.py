"""标注检查网页导出的 CSV：读取与逐点检查。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def label_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "labels.yaml").read_text(encoding="utf-8"))


def parse_csv(path: Path) -> pd.DataFrame:
    """读取网页导出的标注 CSV（point_id, lon, lat, label, confidence, imagery_date, note, ...）。"""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["lon"], df["lat"] = df.lon.astype(float), df.lat.astype(float)
    df["confidence"] = df.confidence.astype(str).str.strip()
    return df


def problems(df: pd.DataFrame) -> pd.Series:
    """逐点问题说明，空字符串表示通过：标签合法、置信度 1–3、有判读依据、有坐标。"""
    labels = set(label_config()["labels"])

    def one(r) -> str:
        p = []
        if r.label not in labels:
            p.append("label")
        if r.confidence not in {"1", "2", "3"}:
            p.append("confidence")
        if not r.imagery_date:
            p.append("imagery_date")
        if pd.isna(r.lon) or pd.isna(r.lat):
            p.append("coords")
        return ",".join(p)

    return df.apply(one, axis=1)
