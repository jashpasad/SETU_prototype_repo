import sys, time; sys.path.insert(0, '.')
from setu.config import SimConfig
from setu.simulate import run, summarise
cfg = SimConfig(days=10)
t0=time.time(); res = run(cfg, progress=False); el=time.time()-t0
s = summarise(res)
print(f"elapsed {el:.1f}s for {cfg.days} days -> {el/cfg.days:.2f} s/day")
for k,v in s.items(): print(f"  {k}: {v}")
