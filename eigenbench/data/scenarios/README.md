# Evaluation scenarios (not included)

The 290 evaluation scenarios are real requests submitted by NYC Council staff to the
deployed assistant. They are not distributed with this package. Access is subject to
institutional approval (see the top-level README).

Expected file: `nycc_scenarios.json`, a JSON list of 290 strings (one request each), in
the original order. EigenBench reads it through `runs/nycc_final/spec.py`.
