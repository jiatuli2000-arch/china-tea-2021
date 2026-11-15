"""S1 · 为打点包每个点生成 2021 年 Sentinel-2 真彩色小图（冬季 / 夏季各一张）。

共享网页不能加载外部底图，用这两张小图代替，并且正好是 2021 年的影像：
冬季图看是否常绿，夏季图看地块纹理。

用法：python stages/S1_probes/extract_chips.py huangshan_kit_v1
输出：data/interim/S1/<kit>_chips.json  {point_id: {"w": dataURI, "s": dataURI}}
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
YEAR = yaml.safe_load((ROOT / "config/project.yaml").read_text(encoding="utf-8"))["year"]
HALF = 400      # 小图半宽（米）→ 800 m 见方
PX = 80         # 输出像素 → 10 m/像素
VIS = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 1800, "gamma": 1.2}


def composite(start: str, end: str) -> ee.Image:
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start, end)
           .linkCollection(cs, ["cs_cdf"])
           .map(lambda i: i.updateMask(i.select("cs_cdf").gte(0.6))))
    return col.median().visualize(**VIS)


def chip(img: ee.Image, lon: float, lat: float) -> str:
    region = ee.Geometry.Point([lon, lat]).buffer(HALF).bounds()
    url = img.getThumbURL({"region": region, "dimensions": PX, "format": "jpg"})
    data = urllib.request.urlopen(url, timeout=60).read()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def main(kit: str) -> None:
    ee.Initialize(project=os.environ.get("GEE_PROJECT", "ee-jiatuli2000"))
    df = pd.read_csv(ROOT / f"data/interim/S1/{kit}.csv")
    winter = composite(f"{YEAR}-01-01", f"{YEAR}-03-01")   # 1–2 月
    summer = composite(f"{YEAR}-06-01", f"{YEAR}-09-01")   # 6–8 月

    def one(r):
        try:
            return r.point_id, {"w": chip(winter, r.lon, r.lat), "s": chip(summer, r.lon, r.lat)}
        except Exception as e:  # 失败的点如实记录，不静默跳过
            return r.point_id, {"error": str(e)[:200]}

    with ThreadPoolExecutor(8) as ex:
        out = dict(ex.map(one, df.itertuples()))
    bad = [k for k, v in out.items() if "error" in v]
    dst = ROOT / f"data/interim/S1/{kit}_chips.json"
    dst.write_text(json.dumps(out), encoding="utf-8")
    print(f"→ {dst}（{dst.stat().st_size / 1e6:.1f} MB）；失败 {len(bad)} 点 {bad[:5]}")


if __name__ == "__main__":
    main(sys.argv[1])
