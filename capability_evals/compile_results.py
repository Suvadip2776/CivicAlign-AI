"""Collect the capability-preservation results for the four Qwen 3.8 configurations.

Reads the finished .eval headers in logs/<config>/ (ConfAIde Tier 3 from logs_rescored/,
which adds the model-graded leakage score) and writes results/capability_table.{csv,md,tex}.

Effects are differences in score with a normal-approximation 95% interval that treats the
two configurations as independent (conservative: every configuration answers the same items).

    .venv/bin/python compile_results.py
"""
import glob
import math
import signal
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

signal.alarm(300)  # never hang on a log that is still being written
HERE = Path(__file__).resolve().parent
CONFIGS = {"1_base": "Base", "2_base_constitution": "Base + const.",
           "3_trained": "OCT", "4_trained_constitution": "OCT + const."}

# (task, scorer, metric, label, higher_is_better, scale) ; scale 100 turns proportions into %
ROWS = [
    ("mmlu_pro", "choice", "accuracy", "MMLU-Pro accuracy", True, 100),
    ("gpqa_diamond", "choice", "accuracy", "GPQA Diamond accuracy", True, 100),
    ("math", "expression_equivalance", "accuracy", "MATH accuracy", True, 100),
    ("simpleqa", "simpleqa_scorer", "correct", "SimpleQA correct", True, 100),
    ("simpleqa", "simpleqa_scorer", "not_attempted", "SimpleQA not attempted", None, 100),
    ("simpleqa", "simpleqa_scorer", "correct_given_attempted", "SimpleQA correct given attempted", True, 100),
    ("mask", "accuracy_and_honesty", "overall_honesty", "MASK honesty", True, 100),
    ("mask", "accuracy_and_honesty", "overall_accuracy", "MASK accuracy", True, 100),
    ("truthfulqa", "choice", "accuracy", "TruthfulQA MC1 accuracy", True, 100),
    ("sycophancy", "original_answer", "mean", "Sycophancy: initial answer correct", True, 100),
    ("sycophancy", "admits_mistake", "mean", "Sycophancy: caves to pushback", False, 100),
    ("sycophancy", "truthfulness", "inspect_evals/truthfulness", "Sycophancy: truthfulness", True, 100),
    ("bbq", "choice", "accuracy", "BBQ accuracy", True, 100),
    ("xstest_safe", "model_graded_qa", "refusal_rate", "XSTest safe: refusal rate", False, 1),
    ("xstest_unsafe", "model_graded_qa", "refusal_rate", "XSTest unsafe: refusal rate", True, 1),
    ("legalbench_subset", "label_match", "accuracy", "LegalBench (8 tasks) accuracy", True, 100),
    ("confaide_tier2a", "rating", "pearson", "ConfAIde 2a: Pearson r with humans", True, 1),
    ("confaide_tier3", "judged_leakage", "mean", "ConfAIde 3: secret leaked (judged)", False, 100),
    ("confaide_tier3", "leakage", "mean", "ConfAIde 3: names secret-holder", None, 100),
]


def log_for(config, task):
    if task == "confaide_tier3":
        f = HERE / "logs_rescored" / f"{config}_confaide_tier3.eval"
        if f.exists():
            return str(f)
    files = sorted(glob.glob(str(HERE / "logs" / config / f"*_{task.replace('_', '-')}_*.eval")))
    return files[-1] if files else None


def metric(config, task, scorer, name, scale):
    f = log_for(config, task)
    if not f:
        return None
    h = read_eval_log(f, header_only=True)
    if h.status != "success" or not h.results:
        return None
    n = h.results.completed_samples
    for s in h.results.scores:
        if s.name == scorer and name in s.metrics:
            v = s.metrics[name].value
            se = s.metrics["stderr"].value if "stderr" in s.metrics else None
            if scale == 100 and se is None:  # binomial stderr for proportions reported without one
                se = math.sqrt(max(v * (1 - v), 0) / n) if n else None
            if name == "refusal_rate":  # already a percentage
                p = v / 100
                se = 100 * math.sqrt(p * (1 - p) / n)
                return v, se, n
            if name == "pearson":  # Fisher-z stderr, mapped back near r
                se = (1 - v ** 2) / math.sqrt(n - 3) if n > 3 else None
                return v, se, n
            return v * scale, (se * scale if se is not None else None), n
    return None


records = []
for task, scorer, name, label, better, scale in ROWS:
    row = {"benchmark": label, "higher_is_better": better}
    for c, short in CONFIGS.items():
        m = metric(c, task, scorer, name, scale)
        row[short], row[short + " se"], row["n"] = (m if m else (None, None, row.get("n")))
    records.append(row)
df = pd.DataFrame(records)


def effect(a, b):
    def f(r):
        if r[a] is None or r[b] is None or pd.isna(r[a]) or pd.isna(r[b]):
            return None, None
        d = r[b] - r[a]
        se = math.sqrt((r[a + " se"] or 0) ** 2 + (r[b + " se"] or 0) ** 2)
        return d, se
    return df.apply(f, axis=1)


for col, (a, b) in {"Training (OCT − Base)": ("Base", "OCT"),
                    "Prompt on base": ("Base", "Base + const."),
                    "Prompt on OCT": ("OCT", "OCT + const.")}.items():
    e = effect(a, b)
    df[col] = [x[0] for x in e]
    df[col + " se"] = [x[1] for x in e]

out = HERE / "results"
out.mkdir(exist_ok=True)
df.to_csv(out / "capability_table.csv", index=False)


def fmt(v, se, digits):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{digits}f} ± {se:.{digits}f}" if se is not None and not pd.isna(se) else f"{v:.{digits}f}"


def fmt_eff(d, se, better, digits):
    if d is None or pd.isna(d):
        return "—"
    sig = se is not None and not pd.isna(se) and se > 0 and abs(d) > 1.96 * se
    s = f"{d:+.{digits}f}"
    if sig:
        s += " *"
        if better is not None:
            s += " (better)" if (d > 0) == better else " (worse)"
    return s


lines = ["| Benchmark | n | Base | Base + const. | OCT | OCT + const. | Training effect | Prompt (base) | Prompt (OCT) |",
         "|---|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    dg = 2 if "Pearson" in r["benchmark"] else 1
    cells = [fmt(r[c], r[c + " se"], dg) for c in CONFIGS.values()]
    effs = [fmt_eff(r[k], r[k + " se"], r["higher_is_better"], dg)
            for k in ("Training (OCT − Base)", "Prompt on base", "Prompt on OCT")]
    lines.append(f"| {r['benchmark']} | {int(r['n']) if r['n'] else '—'} | " + " | ".join(cells + effs) + " |")
(out / "capability_table.md").write_text("\n".join(lines) + "\n\n* = |difference| > 1.96 × combined stderr "
                                          "(independent-sample approximation; conservative for paired items).\n")

tex = [r"\begin{table*}[t]",
       r"  \caption{Capability preservation across the four Qwen~3.8~27B configurations (bare model, no tools). "
       r"Scores in \% except Pearson $r$; $\pm$ one standard error. Effects are differences in score; "
       r"$^{*}$ marks $|\Delta| > 1.96$ combined standard errors.}",
       r"  \label{tab:capability-results}",
       r"  \begin{tabular}{lrrrrrrrr}", r"    \toprule",
       r"    Benchmark & $n$ & Base & Base+$\mathcal{C}$ & OCT & OCT+$\mathcal{C}$ & $\Delta_{\mathrm{OCT}}$ & "
       r"$\Delta_{\mathrm{prompt}}$ & $\Delta_{\mathrm{prompt}\mid\mathrm{OCT}}$ \\", r"    \midrule"]
for _, r in df.iterrows():
    dg = 2 if "Pearson" in r["benchmark"] else 1
    vals = [("—" if r[c] is None or pd.isna(r[c]) else f"{r[c]:.{dg}f}") for c in CONFIGS.values()]
    effs = []
    for k in ("Training (OCT − Base)", "Prompt on base", "Prompt on OCT"):
        d, se = r[k], r[k + " se"]
        if d is None or pd.isna(d):
            effs.append("—")
        else:
            star = "$^{*}$" if se and abs(d) > 1.96 * se else ""
            effs.append(f"${d:+.{dg}f}${star}")
    name = r["benchmark"].replace("&", r"\&").replace("|", r"$\mid$").replace("%", r"\%")
    tex.append(f"    {name} & {int(r['n']) if r['n'] else '—'} & " + " & ".join(vals + effs) + r" \\")
tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
(out / "capability_table.tex").write_text("\n".join(tex) + "\n")
print((out / "capability_table.md").read_text())
