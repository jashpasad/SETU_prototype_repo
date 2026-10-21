import sys, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from setu.config import SimConfig
from setu.simulate import run, summarise
cfg = SimConfig(days=365)
res = run(cfg, enforce_fairness=False, progress=False)
s = summarise(res); led = res["ledger"].summary()
g = res["daily_gini"][30:]
out = {
 "control_gini_mean_steady": round(float(g.mean()),3),
 "control_gini_p95_steady": round(float(np.percentile(g,95)),3),
 "control_gini_within_limit_pct": round(float((g<=0.25).mean()*100),1),
 "control_burden_share_max_pct": round(led["burden_share_max_pct"],1),
 "control_full_outage_h_per_day": s["setu_full_outage_h_per_day"],
 "control_unserved_critical_mwh_yr": s["setu_unserved_critical_mwh_yr"],
 "control_curtailed_kwh_max": round(led["curtailed_kwh_max"],1),
}
p = Path("results/summary.json"); d = json.loads(p.read_text()); d.update(out)
p.write_text(json.dumps(d, indent=2))
np.savez_compressed("results/traces_control.npz", burden_share=res["ledger"].burden_share(), daily_gini=res["daily_gini"])
for k,v in out.items(): print(f"  {k:<38} {v}")
