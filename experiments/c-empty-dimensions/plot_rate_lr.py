"""Overlay pythia-2.8b's learning-rate schedule on the rate of new input directions.

Schedule from EleutherAI/pythia models/2.8B/pythia-2.8b.yml at commit
a19eecb807ec2c79a39ebf18108816e6ffffc1d5: lr 1.6e-4, min_lr 1.6e-5, cosine decay over 143,000 steps
(lr-decay-iters), warmup 0.01 of the steps.

Directions needed: input table, within 0.01 bits/byte, interpolated between
k steps (results/phase3_checkpoints.json). Step 1000 is left out (the table
is still close to random there).

Writes results/rate_vs_lr.png and prints the per-interval numbers.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LR_MAX, LR_MIN, T, WARMUP = 1.6e-4, 1.6e-5, 143000, 0.01
TOL = 0.01


def lr(t):
    tw = WARMUP * T
    t = np.asarray(t, dtype=float)
    prog = np.clip((t - tw) / (T - tw), 0, 1)
    return np.where(t < tw, LR_MAX * t / tw, LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * prog)))


def k_needed(curve, base):
    pts = [(p["k"], p["bpb"] - base) for p in curve]
    for (k0, e0), (k1, e1) in zip(pts, pts[1:]):
        if e0 > TOL >= e1:
            return k0 + (k1 - k0) * (e0 - TOL) / (e0 - e1)
    return pts[0][0]


rows = json.loads((HERE / "results" / "phase3_checkpoints.json").read_text())
pts = sorted((int(r["label"][4:]), k_needed(r["input"]["curve"], r["baseline_bpb"])) for r in rows)
pts = [p for p in pts if p[0] >= 8000]

steps = np.arange(T + 1)
cum_lr = np.cumsum(lr(steps))  # total step size taken up to each step
print("interval            new dirs/1k steps   mean lr     new dirs per unit of summed lr")
mids, rate, per_lr = [], [], []
for (s0, k0), (s1, k1) in zip(pts, pts[1:]):
    dk = k1 - k0
    mids.append((s0 + s1) / 2)
    rate.append(dk / ((s1 - s0) / 1000))
    summed = cum_lr[s1] - cum_lr[s0]
    per_lr.append(dk / summed)
    print(f"{s0:6d}-{s1:6d}      {rate[-1]:5.2f}             {summed / (s1 - s0):.2e}   {per_lr[-1]:8.1f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
a1.plot(mids, rate, "o-", color="C1", label="new input directions per 1,000 steps")
a1.set_xlabel("training step")
a1.set_ylabel("new directions per 1,000 steps", color="C1")
a1.set_ylim(bottom=0)
b1 = a1.twinx()
b1.plot(steps, lr(steps), color="C0", label="learning rate")
b1.set_ylabel("learning rate", color="C0")
b1.set_ylim(bottom=0)
a1.set_title("pythia-2.8b: rate of new directions vs learning rate")
a2.plot(mids, per_lr, "o-", color="C2")
a2.set_xlabel("training step")
a2.set_ylabel("new directions per unit of summed learning rate")
a2.set_ylim(bottom=0)
a2.set_title("Rate with the schedule divided out")
fig.tight_layout()
fig.savefig(HERE / "results" / "rate_vs_lr.png", dpi=130)
