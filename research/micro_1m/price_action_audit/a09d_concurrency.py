"""Audit: how many positions the uncapped event study holds at the same time (all six perpetuals together)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
import a09_overlap as ov
out = {}
for segment in ("design", "validation"):
    ev = ov.load(segment)
    e = ev[np.isfinite(ev["g240"])]
    m = e["minute"].to_numpy()
    lo, hi = m.min(), m.max() + 242
    delta = np.zeros(hi - lo + 2)
    np.add.at(delta, m + 1 - lo, 1)          # open at t+1
    np.add.at(delta, m + 241 - lo, -1)       # close at t+1+240
    conc = np.cumsum(delta)
    at_entry = conc[m + 1 - lo]               # positions open right after this entry (all coins)
    long_net = np.zeros(hi - lo + 2)
    np.add.at(long_net, m + 1 - lo, e["dir"].to_numpy())
    np.add.at(long_net, m + 241 - lo, -e["dir"].to_numpy())
    net = np.cumsum(long_net)
    out[segment] = dict(events=int(len(e)), events_per_day=float(len(e) / ((hi - lo) / 1440)),
                        open_positions_time_average=float(conc.mean()), open_positions_max=int(conc.max()),
                        open_positions_at_entry_median=float(np.median(at_entry)), p90=float(np.percentile(at_entry, 90)), p99=float(np.percentile(at_entry, 99)),
                        share_of_time_flat=float((conc == 0).mean()), net_long_positions_max=int(net.max()), net_short_positions_max=int(-net.min()),
                        share_of_total_gross_earned_while_more_than_10_open=float(e.loc[at_entry > 10, "g240"].sum() / e["g240"].sum()),
                        share_of_events_with_more_than_10_open=float((at_entry > 10).mean()))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(al.OUT, "concurrency.json"), "w"), indent=1)
