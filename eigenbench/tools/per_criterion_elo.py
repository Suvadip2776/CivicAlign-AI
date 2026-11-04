"""Per-criterion EigenBench Elo for one run, with the run's own pipeline and bootstrap.

For each constitutional criterion k, every judgment keeps only its criterion-k rating and
the unmodified EigenBench direct-rating pipeline (z-score softmax, EigenTrust, Elo) is run
on that single criterion, plus the same scenario bootstrap (100 resamples, seed 42).

    EIGENBENCH_DIR=EigenBench python eigenbench/tools/per_criterion_elo.py EigenBench/runs/nycc_final
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("EIGENBENCH_DIR", "EigenBench"))
from pipeline.train.direct_analysis import _resample_scenarios  # noqa: E402
from pipeline.train.train import eigentrust_to_elo  # noqa: E402
from pipeline.trust.direct_rating import build_direct_trust  # noqa: E402

RUN = Path(sys.argv[1])
cfg = json.load(open(RUN / "direct_rating/analysis_config.json"))
labels, n_crit = cfg["model_order"], cfg["num_criteria"]
records = [json.loads(l) for l in open(RUN / "evaluations.jsonl")]
records = [r for r in records if r.get("record_type") == "direct_rating"]
NAMES = [r["criterion"] for r in records[0]["ratings"]]


def elo(recs):
    res = build_direct_trust(recs, num_models=len(labels), num_criteria=1,
                             include_self=cfg["include_self"], allow_sparse=True, normalization=cfg["normalization"],
                             softmax_temperature=cfg["softmax_temperature"],
                             scale_min=cfg["scale_min"], scale_max=cfg["scale_max"],
                             eigentrust_alpha=cfg["eigentrust_alpha"])
    return eigentrust_to_elo(res.eigentrust_scores, len(labels))


out = {}
for k in range(n_crit):
    recs = [{**r, "ratings": [{**x, "criterion_index": 0} for x in r["ratings"] if x["criterion_index"] == k]}
            for r in records]
    point = elo(recs)
    rng = random.Random(42)
    boot = np.array([elo(_resample_scenarios(recs, rng)) for _ in range(100)])
    order = np.argsort(-point)
    out[str(k + 1)] = {"criterion": NAMES[k], "arms": {
        labels[i]: {"elo": float(point[i]), "rank": int(np.where(order == i)[0][0] + 1),
                    "boot_mean": float(boot[:, i].mean()),
                    "ci": [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))]}
        for i in range(len(labels))}}
    print(f"criterion {k + 1} done", flush=True)
(RUN / "direct_rating/per_criterion_elo.json").write_text(json.dumps(out, indent=1))
print("wrote", RUN / "direct_rating/per_criterion_elo.json")
