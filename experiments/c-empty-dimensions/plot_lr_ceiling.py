"""Learning rate as a ceiling on the rate of new input directions (pythia-2.8b).

Puts the learning-rate schedule on the same scale as the rate of new
directions: the schedule is multiplied by the smallest factor that keeps it
on or above every measured interval (comparing each interval's rate with the
schedule's mean over that interval). Where the scaled curve touches the
points, the learning rate is what holds the rate down; where it sits far
above, something else does.

Uses the schedule and the interpolated directions from plot_rate_lr.py.
Writes results/lr_ceiling.png.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_rate_lr import HERE, T, lr, pts

steps = np.arange(T + 1)
sched = lr(steps)
cum = np.cumsum(sched)
mids, rate, mean_lr = [], [], []
for (s0, k0), (s1, k1) in zip(pts, pts[1:]):
    mids.append((s0 + s1) / 2)
    rate.append((k1 - k0) / ((s1 - s0) / 1000))
    mean_lr.append((cum[s1] - cum[s0]) / (s1 - s0))
scale = max(r / m for r, m in zip(rate, mean_lr))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(steps, sched * scale, color="C0", label="learning rate, scaled to sit on top")
ax.plot(mids, rate, "o-", color="C1", label="new input directions per 1,000 steps")
for m, r, l in zip(mids, rate, mean_lr):
    ax.plot([m, m], [r, l * scale], color="grey", lw=1, ls=":")
ax.set_xlabel("training step")
ax.set_ylabel("new directions per 1,000 steps")
ax.set_ylim(bottom=0)
ax.set_title("pythia-2.8b: does the learning rate cap the rate of new directions?")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "results" / "lr_ceiling.png", dpi=130)
for m, r, l in zip(mids, rate, mean_lr):
    print(f"step {m:7.0f}: rate {r:4.2f}, ceiling {l * scale:5.2f}, rate/ceiling {r / (l * scale):.2f}")
