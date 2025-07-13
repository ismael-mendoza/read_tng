#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

TNG_H = 0.6774  # from website
FIG_DIR = Path("/home/imendoza/workspace/read_tng/plots/diagnostics")

# def _convert_tng_mass(m):
#    """Convert TNG mass to log10(Msun)."""
#    return np.where(m > 0, np.log10(m * 1e10 / TNG_H), 0)

with open('tng100_trees.npz', 'rb') as f:
    npz_data = np.load(f)

    bbar = npz_data['bbar']
    bdmo = npz_data['bdmo']
    tbar = npz_data['tbar']
    tdmo = npz_data['tdmo']
    matches = npz_data['matches']

print("Finished loading data...")

print("Number of matched hydro subhaloes:", len(matches[matches!=-1]))
print("Percentage of matched hydro subhaloes", len(matches[matches!=-1]) / len(matches))

# histogram
fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.hist(np.log10(bbar['mpeak_pre']), bins=21)
fig.savefig(FIG_DIR / "mpeak_pre_bar_hist.png")

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.hist(np.log10(bdmo['mpeak_pre']), bins=21)
fig.savefig(FIG_DIR / "mpeak_pre_dmo_hist.png")


# sanity check phil suggested

# baryons
mask = (tbar['mdm'][:, -1] > 0) & (bbar['mpeak'] > 0) & (bbar['mpeak_pre'] > 0) & (~bbar['is_err'])
m_now_bar = tbar['mdm'][:, -1][mask]
mpeak_bar = bbar['mpeak'][mask]
mpeak_pre_bar = bbar['mpeak_pre'][mask]

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(np.log10(mpeak_pre_bar), mpeak_pre_bar / m_now_bar, 'o', alpha=0.1, markersize=2)
ax.set_xlabel("m_peak_pre (log10 Msun)")
ax.set_ylabel("m_peak_pre / m_now")
ax.set_yscale("log")
fig.savefig(FIG_DIR / "mpeak_pre_bar_ratio.png")

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(np.log10(mpeak_bar), mpeak_bar / m_now_bar, 'o', alpha=0.1, markersize=2)
ax.set_xlabel("m_peak (log10 Msun)")
ax.set_ylabel("mpeak / m_now")
ax.set_yscale("log")
fig.savefig(FIG_DIR / "mpeak_bar_ratio.png")


# dmo
mask = (tdmo['mdm'][:, -1] > 0) & (bdmo['mpeak'] > 0) & (bdmo['mpeak_pre'] > 0) & (~bdmo['is_err'])
m_now = tdmo['mdm'][:, -1][mask]
mpeak = bdmo['mpeak'][mask]
mpeak_pre = bdmo['mpeak_pre'][mask]

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(np.log10(mpeak_pre), mpeak_pre / m_now, 'o', alpha=0.1, markersize=2)
ax.set_xlabel("m_peak_pre (log10 Msun)")
ax.set_ylabel("m_peak_pre / m_now")
ax.set_yscale("log")
fig.savefig(FIG_DIR / "mpeak_pre_dmo_ratio.png")

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(np.log10(mpeak), mpeak / m_now, 'o', alpha=0.1, markersize=2)
ax.set_xlabel("m_peak (log10 Msun)")
ax.set_ylabel("mpeak / m_now")
ax.set_yscale("log")
fig.savefig(FIG_DIR / "mpeak_dmo_ratio.png")
