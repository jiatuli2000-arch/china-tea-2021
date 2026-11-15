"""S1 · 把标注完成后导出的 KML 入库为探针表。

用法：python stages/S1_probes/ingest.py 导出.csv（网页导出）或 导出.kml（Google Earth 导出） --zone jiangnan --area huangshan --province 安徽 --annotator xxx

- 有问题的点（标签不合法、缺置信度、缺影像日期）不入库，列在 *_problems.csv 中，改完重新导出即可。
- 输出：data/raw/probes/<zone>__<area>.csv（不覆盖已有文件，存在就报错）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
from kml_io import parse_csv, parse_kml  # noqa: E402

COLS = ["point_id", "lon", "lat", "label", "confidence", "imagery_source", "imagery_date",
        "zone", "province", "area", "annotator", "note"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--zone", required=True, choices=["jiangbei", "jiangnan", "huanan", "xinan"])
    ap.add_argument("--area", required=True)
    ap.add_argument("--province", required=True)
    ap.add_argument("--annotator", required=True)
    a = ap.parse_args()

    out = ROOT / f"data/raw/probes/{a.zone}__{a.area}.csv"
    if out.exists():
        sys.exit(f"已存在 {out}，不覆盖。若是新一轮标注请换 --area（如 huangshan_r2）。")

    df = parse_csv(a.src) if a.src.suffix.lower() == ".csv" else parse_kml(a.src)
    bad = df[df.problem != ""]
    ok = df[df.problem == ""].copy()
    ok["imagery_source"] = "Google Earth Web"
    if "note" not in ok:
        ok["note"] = ""
    ok["zone"], ok["province"], ok["area"], ok["annotator"] = a.zone, a.province, a.area, a.annotator
    ok["confidence"] = ok.confidence.astype(int)

    out.parent.mkdir(parents=True, exist_ok=True)
    ok[COLS].to_csv(out, index=False, encoding="utf-8-sig")
    if len(bad):
        bad.to_csv(out.with_name(out.stem + "_problems.csv"), index=False, encoding="utf-8-sig")
    print(f"入库 {len(ok)} 点 → {out}；待修正 {len(bad)} 点")
    print(ok.label.value_counts().to_string())


if __name__ == "__main__":
    main()
