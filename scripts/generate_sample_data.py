import json, math, random
from pathlib import Path

values = [100 + 8 * math.sin(i / 4) + random.uniform(-2, 2) for i in range(80)]
values[25] = 150
values[55] = 62
payload = {"metric_name": "api_latency_ms", "values": [round(v, 2) for v in values], "forecast_horizon": 7}
Path("data").mkdir(exist_ok=True)
Path("data/sample_series.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("Wrote data/sample_series.json")
