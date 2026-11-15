"""S1 · 生成多人协作标注页（发布为 claude.ai Artifact，标注存入页面数据库）。

用法：python stages/S1_probes/build_shared.py huangshan_kit_v1
输入：<kit>.csv、<kit>_ts.json、<kit>_chips.json（均在 data/interim/S1/）
输出：data/interim/S1/<kit>_shared.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"huangshan": "黄山"}


def main(kit: str) -> None:
    d = ROOT / "data/interim/S1"
    k = pd.read_csv(d / f"{kit}.csv")[["point_id", "lon", "lat", "label", "source", "desc"]].fillna("")
    ts = json.loads((d / f"{kit}_ts.json").read_text(encoding="utf-8"))
    chips = json.loads((d / f"{kit}_chips.json").read_text(encoding="utf-8"))
    chips = {p: v for p, v in chips.items() if "error" not in v}
    labels = yaml.safe_load((ROOT / "config/labels.yaml").read_text(encoding="utf-8"))["labels"]
    area = kit.split("_")[0]
    title = f"{NAMES.get(area, area)}茶园探针标注"
    html = (ROOT / "scripts/common/shared_labeler_template.html").read_text(encoding="utf-8")
    for key, val in {
        "__TITLE__": title, "__YEAR__": str(ts["year"]), "__KITNAME__": kit,
        "__KIT__": json.dumps(k.to_dict("records"), ensure_ascii=False),
        "__TS__": json.dumps(ts, ensure_ascii=False),
        "__CHIPS__": json.dumps(chips), "__LABELS__": json.dumps(labels),
    }.items():
        html = html.replace(key, val)
    out = d / f"{kit}_shared.html"
    out.write_text(html, encoding="utf-8")
    print(f"→ {out}（{out.stat().st_size / 1e6:.1f} MB）；有小图 {len(chips)}/{len(k)} 点")


if __name__ == "__main__":
    main(sys.argv[1])
