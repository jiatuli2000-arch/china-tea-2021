"""S1 · 黄山打点包 v1：核验 241 个旧判读点 + 补充非茶候选点。

输出：data/interim/S1/huangshan_kit_v1.kml（导入 Google Earth 网页版）
      data/interim/S1/huangshan_kit_v1.csv（同内容表格，便于追溯）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import ee
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
from kml_io import write_kml  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "project.yaml").read_text(encoding="utf-8"))
YEAR = CFG["year"]
SEED = 2021
LEGACY = ROOT / "data/raw/legacy/huangshan2021_manual_points.csv"
OUT = ROOT / "data/interim/S1/huangshan_kit_v1"
BOUNDARY = "projects/project-c7b654cb-9d08-48ed-bcf/assets/01_boundary/study_areas/Huangshan_Boundary"

# 旧标签 → 建议标签（需逐点核验）
LEGACY_MAP = {"茶园": "tea", "常绿非茶树": "?evergreen", "林地待分型": "?", "落叶林": "deciduous_forest"}
# ESA WorldCover 2021 类别 → 补充候选点数量（只作为抽样分层，不作为标签）
ESA_QUOTA = {10: 90, 40: 40, 20: 15, 30: 15}
ESA_NAME = {10: "树木", 20: "灌丛", 30: "草地", 40: "农田"}


def legacy_points() -> pd.DataFrame:
    d = pd.read_csv(LEGACY)
    out = pd.DataFrame({
        "point_id": d.point_id, "lon": d.lon, "lat": d.lat,
        "label": d.class_label.map(LEGACY_MAP).fillna("?"),
        "source": "legacy_huangshan2021",
    })
    out["desc"] = "旧标签=" + d.class_label + "；旧备注=" + d.notes.fillna("") + "；请切到2021前后影像核验，写 img=YYYY-MM"
    return out


def esa_candidates() -> pd.DataFrame:
    assert YEAR == 2021, "ESA WorldCover v200 对应 2021 年"
    region = ee.FeatureCollection(BOUNDARY).geometry()
    esa = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("esa")
    classes, points = list(ESA_QUOTA), list(ESA_QUOTA.values())
    fc = esa.stratifiedSample(
        numPoints=0, classBand="esa", region=region, scale=10, seed=SEED,
        classValues=classes, classPoints=points, geometries=True, tileScale=4,
    )
    feats = fc.getInfo()["features"]
    rows = []
    for i, f in enumerate(feats, 1):
        lon, lat = f["geometry"]["coordinates"]
        c = f["properties"]["esa"]
        rows.append({"point_id": f"HS21N_{i:04d}", "lon": lon, "lat": lat, "label": "?",
                     "source": f"esa{YEAR}_class{c}",
                     "desc": f"新候选点（ESA {YEAR}={ESA_NAME[c]}，仅供参考）；写 img=YYYY-MM"})
    return pd.DataFrame(rows)


def main() -> None:
    ee.Initialize(project=os.environ.get("GEE_PROJECT", "ee-jiatuli2000"))
    df = pd.concat([legacy_points(), esa_candidates()], ignore_index=True)
    assert df.point_id.is_unique
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    write_kml(df, OUT.with_suffix(".kml"), f"黄山打点包 v1（{YEAR}）")
    print(f"{len(df)} 点 → {OUT.with_suffix('.kml')}")
    print(df.source.value_counts().to_string())
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main()
