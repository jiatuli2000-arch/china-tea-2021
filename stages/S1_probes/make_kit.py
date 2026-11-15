"""S1 · 通用打点包生成：在一个县/市内按 2021 年物候规则找各类候选点，盲标。

用法（福建安溪）：
  python stages/S1_probes/make_kit.py --kit anxi_kit_v1 --prefix AX21 --county 安溪县 \
      --n tea=150 --n evergreen=90 --n deciduous=60 --n cropland=70 --pool 8000

候选规则（只用于找候选；阈值参考黄山探针点的类别分位数，见 data/interim/S2/e01_features.csv）：
  tea        ESA ∈ {树木, 灌丛, 草地, 农田}，冬季 NDVI > 0.6，夏季 NDVI < 0.78，坡度 > 2°
  evergreen  ESA = 树木，冬季 NDVI > 0.7，夏季 NDVI ≥ 0.80
  deciduous  ESA = 树木，冬季 NDVI < 0.5，夏季 NDVI > 0.75
  cropland   ESA = 农田，全年 NDVI P90 − P10 > 0.3
规则命中多类时按 deciduous > cropland > tea > evergreen 取一类。

盲标：打点包里的 source 只写 "cand"，标签留空，描述只给实测值，不写候选类型；
候选类型另存 data/interim/S1/<kit>_candidates.csv，标完后用来统计各规则的命中率。
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
COUNTIES = "projects/project-c7b654cb-9d08-48ed-bcf/assets/01_boundary/china_provinces/china_AMD3_Areas"
MIN_DIST_M, SEED = 200, 2021
ORDER = ["deciduous", "cropland", "tea", "evergreen"]


def rule(p: dict) -> str | None:
    esa, w, s, amp, slope = (p.get(k) for k in ("esa", "w", "s", "amp", "slope"))
    if None in (esa, w, s):
        return None
    hits = set()
    if esa == 10 and w < 0.5 and s > 0.75:
        hits.add("deciduous")
    if esa == 40 and amp is not None and amp > 0.3:
        hits.add("cropland")
    if esa in (10, 20, 30, 40) and w > 0.6 and s < 0.78 and (slope or 0) > 2:
        hits.add("tea")
    if esa == 10 and w > 0.7 and s >= 0.80:
        hits.add("evergreen")
    return next((c for c in ORDER if c in hits), None)


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
    ap.add_argument("--county", required=True, help="县级名称，如 安溪县")
    ap.add_argument("--n", action="append", default=[], metavar="类别=数量")
    ap.add_argument("--pool", type=int, default=6000)
    ap.add_argument("--exclude", action="append", default=[])
    a = ap.parse_args()
    want = {k: int(v) for k, v in (x.split("=") for x in a.n)}
    assert set(want) <= set(ORDER), want
    assert YEAR == 2021, "ESA WorldCover v200 对应 2021 年"
    out = ROOT / f"data/interim/S1/{a.kit}.csv"
    if out.exists():
        raise SystemExit(f"已存在 {out}，不覆盖")

    ee.Initialize(project=os.environ.get("GEE_PROJECT", "ee-jiatuli2000"))
    fc = ee.FeatureCollection(COUNTIES).filter(ee.Filter.eq("name", a.county))
    assert fc.size().getInfo() == 1, f"县名 {a.county} 不唯一或不存在"
    region = fc.geometry()
    esa = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("esa")
    w = ndvi_col(f"{YEAR}-01-01", f"{YEAR}-03-01", region).median().rename("w")
    s = ndvi_col(f"{YEAR}-06-01", f"{YEAR}-09-01", region).median().rename("s")
    pct = ndvi_col(f"{YEAR}-01-01", f"{YEAR + 1}-01-01", region).reduce(ee.Reducer.percentile([10, 90]))
    amp = pct.select("ndvi_p90").subtract(pct.select("ndvi_p10")).rename("amp")
    demc = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    ter = ee.Terrain.products(demc.mosaic().setDefaultProjection(demc.first().projection())).select(["slope", "aspect"])
    stack = ee.Image.cat([esa, w, s, amp, ter])

    pts = ee.FeatureCollection.randomPoints(region, a.pool, SEED)
    feats = []
    for i in range(0, a.pool, 500):
        feats += stack.sampleRegions(ee.FeatureCollection(pts.toList(500, i)), scale=10,
                                     geometries=True, tileScale=4).getInfo()["features"]
        print(f"取值 {min(i + 500, a.pool)}/{a.pool}", flush=True)

    taken = [(r.lon, r.lat) for f in a.exclude for r in pd.read_csv(ROOT / f).itertuples()]
    count = {k: 0 for k in want}
    rows, cands = [], []
    for f in feats:
        p = f["properties"]
        c = rule(p)
        lonlat = tuple(f["geometry"]["coordinates"])
        if c not in want or count[c] >= want[c] or any(dist_m(lonlat, t) < MIN_DIST_M for t in taken):
            continue
        taken.append(lonlat)
        count[c] += 1
        pid = f"{a.prefix}_{len(rows) + 1:04d}"
        rows.append({"point_id": pid, "lon": lonlat[0], "lat": lonlat[1], "label": "?", "source": "cand",
                     "desc": f"冬季 NDVI {p['w']:.2f}，夏季 {p['s']:.2f}，全年振幅 {p.get('amp', float('nan')):.2f}；"
                             f"坡度 {p.get('slope', float('nan')):.0f}°，坡向 {p.get('aspect', float('nan')):.0f}°"})
        cands.append({"point_id": pid, "cand": c, **{k: p.get(k) for k in ("esa", "w", "s", "amp", "slope", "aspect")}})
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)  # 打乱顺序，避免同类扎堆
    df.to_csv(out, index=False, encoding="utf-8-sig")
    pd.DataFrame(cands).to_csv(ROOT / f"data/interim/S1/{a.kit}_candidates.csv", index=False)
    print(f"{len(df)} 点 → {out}")
    print({k: f"{count[k]}/{want[k]}" for k in want})


if __name__ == "__main__":
    main()
