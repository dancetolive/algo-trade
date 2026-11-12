"""Audit: exact figures for the coarse groups quoted in the verdict (fresh / one / two or more / four or more earlier events)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
import a09_overlap as ov
out = {}
for segment in ("design", "validation"):
    ev = ov.load(segment)
    for h in (240, 60):
        e = ev[np.isfinite(ev[f"g{h}"]) & np.isfinite(ev[f"b{h}"])].reset_index(drop=True)
        tot = float(e[f"g{h}"].sum())
        for name, mask in (("no earlier event", e["n_prior"] == 0), ("0 or 1 earlier", e["n_prior"] <= 1), ("2 or more earlier", e["n_prior"] >= 2),
                           ("4 or more earlier", e["n_prior"] >= 4)):
            s = e[mask]
            r = ov.summarise(s, h)
            out[f"{segment}@{h}|{name}"] = dict(events=r["events"], share_events=round(len(s) / len(e), 3), gross=round(r["gross"], 2), t=round(r["t_gross"], 2),
                                               t_week=round(r["t_gross_week"], 2), share_total_gross=round(float(s[f"g{h}"].sum()) / tot, 3),
                                               net_taker=round(r["net_taker"], 2), maker_net_4bp=round(r["maker_net_4bp"], 2), t_maker=round(r["t_maker_net_4bp"], 2))
print(pd.DataFrame(out).T.to_string())
json.dump(out, open(os.path.join(al.OUT, "overlap_groups.json"), "w"), indent=1)
