"""S1 · 生成时序辅助网页（单文件，数据内嵌，双击即可打开）。

用法：python stages/S1_probes/build_viewer.py huangshan_kit_v1
输入：data/interim/S1/<kit>.csv、data/interim/S1/<kit>_ts.json
输出：data/interim/S1/<kit>_viewer.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main(kit: str) -> None:
    d = ROOT / "data/interim/S1"
    k = pd.read_csv(d / f"{kit}.csv")[["point_id", "lon", "lat", "label", "source", "desc"]].fillna("")
    ts = json.loads((d / f"{kit}_ts.json").read_text(encoding="utf-8"))
    labels = yaml.safe_load((ROOT / "config/labels.yaml").read_text(encoding="utf-8"))["labels"]
    html = (ROOT / "scripts/common/ts_viewer_template.html").read_text(encoding="utf-8")
    html = (html.replace("__TITLE__", f"{kit} · 时序辅助")
                .replace("__KIT__", json.dumps(k.to_dict("records"), ensure_ascii=False))
                .replace("__TS__", json.dumps(ts, ensure_ascii=False))
                .replace("__LABELS__", json.dumps(labels)))
    out = d / f"{kit}_viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"→ {out}（{out.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main(sys.argv[1])
