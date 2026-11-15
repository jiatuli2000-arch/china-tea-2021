# data/（不入 Git）

- `raw/`：原始输入（GPS 点、目视解译点、边界、年鉴表）。只读，任何脚本不得改写。
- `interim/`：各阶段中间产物。
- `output/`：冻结的正式成果。

每新增一份数据，必须在 `ledger/data_manifest.csv` 登记来源、日期、sha256。
