import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, ".")
import asyncio
import time

print("Importing main...", flush=True)
t0 = time.time()
from main import (
    get_districts, get_forecast, get_district_detail, get_timeseries,
    get_alerts, get_freshness, get_outbreaks, get_stations,
    get_hotspots, get_drivers, post_whatif, get_metrics,
    get_predlog, export_csv, WhatIfRequest
)
print(f"Import finished in {round(time.time() - t0, 2)}s", flush=True)

async def test_all():
    t_start = time.time()
    
    print("Testing 1: get_districts...", flush=True)
    d = await get_districts("2025-W43")
    print(f"  -> OK: {d['type']} with {len(d['features'])} features", flush=True)

    print("Testing 2: get_forecast...", flush=True)
    f = await get_forecast("2025-W43")
    print(f"  -> OK: {len(f['forecasts'])} forecasts | Uncertainty: {f['uncertainty_metric']}", flush=True)

    print("Testing 3: get_district_detail...", flush=True)
    dt = await get_district_detail("Ernakulam", "2025-W43")
    print(f"  -> OK: {dt['district']} | Pred: {dt['predicted_cases']} | Tier: {dt['tier']}", flush=True)

    print("Testing 4: get_timeseries...", flush=True)
    ts = await get_timeseries("Ernakulam", 12)
    print(f"  -> OK: {ts['weeks_count']} historical points", flush=True)

    print("Testing 5: get_alerts...", flush=True)
    al = await get_alerts("2025-W43")
    print(f"  -> OK: {al['alerts_count']} alerts", flush=True)

    print("Testing 6: get_freshness...", flush=True)
    fr = await get_freshness()
    print(f"  -> OK: {len(fr['sources'])} sources checked", flush=True)

    print("Testing 7: get_outbreaks...", flush=True)
    ob = await get_outbreaks("2025-W43")
    print(f"  -> OK: {len(ob['districts'])} districts reported", flush=True)

    print("Testing 8: get_stations...", flush=True)
    st = await get_stations("Ernakulam")
    print(f"  -> OK: {st['stations_count']} CPCB stations in Ernakulam", flush=True)

    print("Testing 9: get_hotspots (Ernakulam >= 5 stations)...", flush=True)
    hs = await get_hotspots("Ernakulam", "2025-W43")
    print(f"  -> OK: status={hs['status']}, method={hs.get('method')}, {len(hs['hotspots'])} points", flush=True)

    print("Testing 10: get_hotspots (Idukki < 3 fallback)...", flush=True)
    hs_idukki = await get_hotspots("Idukki", "2025-W43")
    print(f"  -> OK: status={hs_idukki['status']} | Msg: {hs_idukki['message']}", flush=True)

    print("Testing 11: get_drivers...", flush=True)
    drv = await get_drivers("Ernakulam", "2025-W43")
    print(f"  -> OK: {len(drv['top_3_drivers'])} top SHAP drivers", flush=True)

    print("Testing 12: post_whatif...", flush=True)
    wi = await post_whatif(WhatIfRequest(district="Ernakulam", week="2025-W43", rainfall_shock_mm=100.0))
    print(f"  -> OK: sim cases={wi['simulated_forecast']['predicted_cases']}, tier={wi['simulated_forecast']['tier']}", flush=True)

    print("Testing 13: get_metrics...", flush=True)
    met = await get_metrics()
    print(f"  -> OK: {len(met)} models in benchmark", flush=True)

    print("Testing 14: get_predlog...", flush=True)
    pl = await get_predlog("Ernakulam")
    print(f"  -> OK: {pl['total_evaluations']} records | Hit rate: {pl['hit_rate_pct']}%", flush=True)

    print("Testing 15: export_csv...", flush=True)
    exp = await export_csv("2025-W43")
    print(f"  -> OK: Media type = {exp.media_type}", flush=True)

    print(f"\nALL 15 ENDPOINTS PASSED IN {round(time.time() - t_start, 2)}s!", flush=True)

if __name__ == "__main__":
    asyncio.run(test_all())
