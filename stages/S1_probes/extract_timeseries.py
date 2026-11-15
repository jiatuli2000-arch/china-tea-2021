"""S1 · 提取打点包各点的年度时序（供时序辅助网页使用）。

用法：python stages/S1_probes/extract_timeseries.py huangshan_kit_v1
输入：data/interim/S1/<kit>.csv（point_id, lon, lat）
输出：data/interim/S1/<kit>_ts.json
  {point_id: {"s2": [[doy, ndvi], ...], "s1": [[doy, vh_db, orbit], ...]}}

- Sentinel-2：S2_SR_HARMONIZED 单景观测，Cloud Score+ (cs_cdf >= 0.6) 去云，不做合成。
- Sentinel-1：S1_GRD IW 模式 VH（dB），升降轨分开记录。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import ee
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
YEAR = yaml.safe_load((ROOT / "config/project.yaml").read_text(encoding="utf-8"))["year"]
CS_THRESH = 0.6
BATCH = 40


def s2_collection(region: ee.Geometry, year: int) -> ee.ImageCollection:
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
          .linkCollection(cs, ["cs_cdf"]))

    def prep(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("v")
        return (ndvi.updateMask(img.select("cs_cdf").gte(CS_THRESH))
                .set("doy", img.date().getRelative("day", "year").add(1)))
    return s2.map(prep)


def s1_collection(region: ee.Geometry, year: int) -> ee.ImageCollection:
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(region).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))

    def prep(img):
        return (img.select("VH").rename("v")
                .set("doy", img.date().getRelative("day", "year").add(1))
                .set("orb", img.get("orbitProperties_pass")))
    return s1.map(prep)


def sample(coll: ee.ImageCollection, pts: ee.FeatureCollection, with_orbit: bool) -> list[dict]:
    def per_img(img):
        fc = img.reduceRegions(pts, ee.Reducer.first().setOutputs(["v"]), scale=10)
        fc = fc.filter(ee.Filter.notNull(["v"])).map(lambda f: f.set("doy", img.get("doy")))
        return fc.map(lambda f: f.set("orb", img.get("orb"))) if with_orbit else fc
    cols = ["point_id", "doy", "v"] + (["orb"] if with_orbit else [])
    flat = coll.map(per_img).flatten().select(cols, None, False)
    return [f["properties"] for f in flat.getInfo()["features"]]


def main(kit: str) -> None:
    ee.Initialize(project=os.environ.get("GEE_PROJECT", "ee-jiatuli2000"))
    df = pd.read_csv(ROOT / f"data/interim/S1/{kit}.csv")
    out: dict[str, dict] = {pid: {"s2": [], "s1": []} for pid in df.point_id}
    for i in range(0, len(df), BATCH):
        part = df.iloc[i:i + BATCH]
        pts = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": r.point_id})
            for r in part.itertuples()])
        region = pts.geometry().bounds().buffer(100)
        for r in sample(s2_collection(region, YEAR), pts, False):
            out[r["point_id"]]["s2"].append([int(r["doy"]), round(r["v"], 4)])
        for r in sample(s1_collection(region, YEAR), pts, True):
            out[r["point_id"]]["s1"].append([int(r["doy"]), round(r["v"], 2), r["orb"][0]])
        print(f"{min(i + BATCH, len(df))}/{len(df)}", flush=True)
    for v in out.values():
        v["s2"].sort()
        v["s1"].sort()
    dst = ROOT / f"data/interim/S1/{kit}_ts.json"
    dst.write_text(json.dumps({"year": YEAR, "points": out}, ensure_ascii=False), encoding="utf-8")
    n2 = [len(v["s2"]) for v in out.values()]
    n1 = [len(v["s1"]) for v in out.values()]
    print(f"→ {dst}\nS2 有效观测/点：中位数 {sorted(n2)[len(n2)//2]}，最少 {min(n2)}；"
          f"S1：中位数 {sorted(n1)[len(n1)//2]}，最少 {min(n1)}")


if __name__ == "__main__":
    main(sys.argv[1])
