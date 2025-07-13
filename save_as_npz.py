#!/usr/bin/env python3


import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


bbar = pickle.load(open("./cache/tng100_bbar.pkl", "rb"))
tbar = pickle.load(open("./cache/tng100_tbar.pkl", "rb"))
bdmo = pickle.load(open("./cache/tng100_bdmo.pkl", "rb"))
tdmo = pickle.load(open("./cache/tng100_tdmo.pkl", "rb") )
matches = pickle.load(open("./cache/matches.pkl", "rb") )

print(bdmo.shape, tdmo.shape, bbar.shape, tbar.shape, matches.shape)


# save all to numpyz
with open('tng100_trees.npz', 'wb') as f:
    np.savez(f, bbar=bbar, tbar=tbar, bdmo=bdmo, tdmo=tdmo, matches=matches)
