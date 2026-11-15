"""S2 · E01b 第一步：逐景估计 L2A NDVI 对 cos i 的残留斜率 β（只用未标注的 ESA 树木像元）。

输出：data/interim/S2/e01b_beta.csv  每景：img, doy, beta, offset, n
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from e01_terrain_extract import BOUNDARY, YEAR, cos_i  # noqa: E402

OUT = ROOT / "data/interim/S2/e01b_beta.csv"


def main() -> None:
    ee.Initialize(project=os.environ.get("GEE_PROJECT", "ee-jiatuli2000"))
    region = ee.FeatureCollection(BOUNDARY).geometry()
    demc = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    ter = ee.Terrain.products(demc.mosaic().setDefaultProjection(demc.first().projection())).select(["slope", "aspect"])
    tree = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(10)
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
           .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01").linkCollection(cs, ["cs_cdf"]))

    def per_image(img):
        st = (img.normalizedDifference(["B8", "B4"]).rename("nd").addBands(cos_i(img, ter))
              .updateMask(tree).updateMask(img.select("cs_cdf").gte(0.6)))
        smp = st.sample(region=region, scale=30, numPixels=3000, seed=7, tileScale=4)
        fit = smp.reduceColumns(ee.Reducer.linearFit(), ["cosi", "nd"])
        return ee.Feature(None, {"img": img.get("system:index"),
                                 "doy": img.date().getRelative("day", "year").add(1),
                                 "beta": fit.get("scale"), "offset": fit.get("offset"), "n": smp.size()})

    ids = col.aggregate_array("system:index").getInfo()
    rows = []
    for i in range(0, len(ids), 20):
        sub = col.filter(ee.Filter.inList("system:index", ids[i:i + 20]))
        rows += [f["properties"] for f in ee.FeatureCollection(sub.map(per_image)).getInfo()["features"]]
        print(f"β {min(i + 20, len(ids))}/{len(ids)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    good = df[df.n >= 300]
    print(f"→ {OUT}：{len(df)} 景，可靠（n≥300）{len(good)} 景；β 中位数 {good.beta.median():+.3f}")


if __name__ == "__main__":
    main()
