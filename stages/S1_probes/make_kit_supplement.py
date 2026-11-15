"""S1 · 补充打点包：针对数量不够的类别，按 2021 年物候特征定向抽候选点。

用法（黄山第二轮，补落叶林和耕地）：
  python stages/S1_probes/make_kit_supplement.py --kit huangshan_kit_r2 --prefix HS21R2 \
      --boundary projects/project-c7b654cb-9d08-48ed-bcf/assets/01_boundary/study_areas/Huangshan_Boundary \
      --exclude data/raw/probes/jiangnan__huangshan.csv --deciduous 70 --cropland 45

候选规则（只用于找候选，标签一律留空，由人判读）：
- 落叶林候选：ESA 2021 = 树木，且冬季（1–2 月）NDVI 中位数 < 0.45、夏季（6–8 月）> 0.75
- 耕地候选：ESA 2021 = 农田，且全年 NDVI 第 90 与第 10 百分位之差 > 0.3
在区内随机撒点（默认 4000 个）逐点取值，按规则筛出候选，再剔除离已有探针点或彼此不到 200 m 的点。
输出：data/interim/S1/<kit>.csv
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import ee
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
YEAR = yaml.safe_load((ROOT / "config/project.yaml").read_text(encoding="utf-8"))["year"]
MIN_DIST_M = 200
SEED = 2021


def ndvi_col(start: str, end: str, region: ee.Geometry) -> ee.ImageCollection:
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(start, end)
            .linkCollection(cs, ["cs_cdf"])
            .map(lambda i: i.normalizedDifference(["B8", "B4"]).rename("ndvi")
                 .updateMask(i.select("cs_cdf").gte(0.6))))


def dist_m(a, b) -> float:
    (lon1, lat1), (lon2, lat2) = a, b
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return 6371000 * math.hypot(x, math.radians(lat2 - lat1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--exclude", action="append", default=[], help="已有探针表，候选点需离它们 200 m 以上")
    ap.add_argument("--deciduous", type=int, default=0)
    ap.add_argument("--cropland", type=int, default=0)
    ap.add_argument("--pool", type=int, default=4000, help="随机撒点数")
    a = ap.parse_args()
    assert YEAR == 2021, "ESA WorldCover v200 对应 2021 年"

    ee.Initialize(project=os.environ.get("GEE_PROJECT", "ee-jiatuli2000"))
    region = ee.FeatureCollection(a.boundary).geometry()
    esa = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    winter = ndvi_col(f"{YEAR}-01-01", f"{YEAR}-03-01", region).median().rename("w")
    summer = ndvi_col(f"{YEAR}-06-01", f"{YEAR}-09-01", region).median().rename("s")
    pct = ndvi_col(f"{YEAR}-01-01", f"{YEAR + 1}-01-01", region).reduce(ee.Reducer.percentile([10, 90]))
    amp = pct.select("ndvi_p90").subtract(pct.select("ndvi_p10")).rename("amp")

    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().setDefaultProjection("EPSG:4326", None, 30)
    terr = ee.Terrain.products(dem).select(["slope", "aspect"])
    stack = ee.Image.cat([esa.rename("esa"), winter, summer, amp, terr])

    # 整区分层抽样在交互模式下会超时：改为随机撒点、逐点取值，再在本地按规则筛选
    want = {1: a.deciduous, 2: a.cropland}
    want = {k: v for k, v in want.items() if v > 0}
    pts = ee.FeatureCollection.randomPoints(region, a.pool, SEED)
    feats = []
    for i in range(0, a.pool, 500):
        chunk = ee.FeatureCollection(pts.toList(500, i))
        feats += stack.sampleRegions(chunk, scale=10, geometries=True, tileScale=4).getInfo()["features"]
        print(f"取值 {min(i + 500, a.pool)}/{a.pool}", flush=True)

    def klass(p) -> int:
        if p.get("esa") == 10 and p.get("w", 1) < 0.45 and p.get("s", 0) > 0.75:
            return 1
        if p.get("esa") == 40 and p.get("amp", 0) > 0.3:
            return 2
        return 0
    for f in feats:
        f["properties"]["cls"] = klass(f["properties"])
    feats = [f for f in feats if f["properties"]["cls"] in want]

    taken = [(r.lon, r.lat) for f in a.exclude for r in pd.read_csv(ROOT / f).itertuples()]
    name = {1: "落叶林候选", 2: "耕地候选"}
    rows, count = [], {k: 0 for k in want}
    for f in feats:
        p = f["properties"]
        c = p["cls"]
        lonlat = tuple(f["geometry"]["coordinates"])
        if count[c] >= want[c] or any(dist_m(lonlat, t) < MIN_DIST_M for t in taken):
            continue
        taken.append(lonlat)
        count[c] += 1
        rows.append({"point_id": f"{a.prefix}_{len(rows) + 1:04d}", "lon": lonlat[0], "lat": lonlat[1],
                     "label": "?", "source": f"supplement_{'deciduous' if c == 1 else 'cropland'}",
                     "desc": f"{name[c]}（依据：冬季 NDVI {p.get('w', float('nan')):.2f}，夏季 {p.get('s', float('nan')):.2f}，"
                             f"全年振幅 {p.get('amp', float('nan')):.2f}；坡度 {p.get('slope', float('nan')):.0f}°，坡向 {p.get('aspect', float('nan')):.0f}°）"})
    df = pd.DataFrame(rows)
    out = ROOT / f"data/interim/S1/{a.kit}.csv"
    if out.exists():
        raise SystemExit(f"已存在 {out}，不覆盖")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(df)} 点 → {out}")
    print({name[k]: f"{count[k]}/{want[k]}" for k in want})


if __name__ == "__main__":
    main()
