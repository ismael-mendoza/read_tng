#!/usr/bin/env python3

import tng as rt
import numpy as np
import pickle

match_file = "/nfs/turbo/lsa-cavestru/imendoza/TNG100_1/matches/subhalo_matching_to_dark.hdf5"
dir_dmo = "/nfs/turbo/lsa-cavestru/kuanwang/TNG_Data/TNG100-1-Dark_SubLink"
dir_bar = "/nfs/turbo/lsa-cavestru/kuanwang/TNG_Data/TNG100-1_SubLink"
CACHE_DIR = "/home/imendoza/workspace/read_tng/cache"

MIN_MASS = 10**(10.0)
MIN_BUFFER = 0.25

def branch_edges(first_prog):
    mids = np.where(first_prog == -1)[0] + 1
    edges = np.zeros(len(mids) + 1, dtype=int)
    edges[1:] = mids
    return edges
    
def naive_mpeak(m, edges):
    mpeak = np.zeros(len(edges) - 1)
    for i in range(len(mpeak)):
        mpeak[i] = np.max(m[edges[i]: edges[i+1]])
    return mpeak

def renumber_edges(edges, b_ok):
    n = edges[1:] - edges[:-1]
    out = np.zeros(np.sum(b_ok)+1, dtype=int)
    out[1:] = np.cumsum(n[b_ok])
    return out
    
def branch_ok_to_tree_ok(edges, b_ok):
    t_ok = np.zeros(edges[-1], dtype=bool)
    for i in range(len(edges) - 1):
        if b_ok[i]:
            t_ok[edges[i]: edges[i+1]] = True
    return t_ok

def reshape_branches(edges, snap, x):
    start, end = edges[:-1], edges[1:]
    n_branch, n_snap = len(edges)-1, 100

    xx = np.zeros((n_branch, n_snap), dtype=x.dtype)
    ok = np.zeros((n_branch, n_snap), dtype=bool)
    tree_to_branch = np.ones(len(x), dtype=np.int64)*-1

    for i in range(n_branch):
        snap_i = snap[start[i]: end[i]]
        x_i = x[start[i]: end[i]]

        xx[i,snap_i] = x_i
        ok[i,snap_i] = True
        tree_to_branch[start[i]: end[i]] = i

    return xx, ok, tree_to_branch

def id_to_index(sub_id, target_id):
    idx = np.arange(len(sub_id), dtype=np.int64)
    return idx + target_id - sub_id

def read_small_reshaped_tree(tree_file, is_hydro=False, read_matches=False):
    # Start by reading in first progenitor ID to work out the edges of
    # the different branches
    first_prog = tree_file.read(["FirstProgenitorID"])[0]
    edges = branch_edges(first_prog)

    # Use the usual, naive Mpeak measure to make a cut on which halos you
    # include. The naive algorithm always overestimates, so you can always
    # cut it down later.
    m = tree_file.read(["SubhaloMass"])[0]
    mpeak_n = naive_mpeak(m, edges)

    # Rough, intial filter to remove most of the junk. b_ok is a filter that
    # can be applied to each branch
    if read_matches:
        b_ok = mpeak_n > MIN_MASS
    else:
        b_ok = mpeak_n > MIN_MASS*MIN_BUFFER
        print(np.sum(b_ok), len(b_ok))
    # convert to a filter that can be applied to each halo
    t_ok = branch_ok_to_tree_ok(edges, b_ok)
    # Change the edge array so that indexes over a smaller, contiguous array
    edges = renumber_edges(edges, b_ok)

    m = m[t_ok]

    # output datastype
    dtype = [("mdm", "f4"), ("mvir", "f4"), ("vmax", "f4"), ("ok", "?"), ("is_sub", "?"),
             ("subfind_id", "i8"), ("first_sub_idx", "i8"), 
             ("match", "i8")]
    if is_hydro:
        dtype.append(("stellar_mass", "f4"))

    t = np.zeros((len(edges)-1, 100), dtype=dtype)
    # Used to figure out what index to put halos in
    snap = tree_file.read(["SnapNum"])[0][t_ok]

    # One by one, call reshape_branches on individual variables, only floating
    # oen full-sized array in memory at a time.
    t["mdm"], t["ok"], tree_to_branch = reshape_branches(edges, snap, m)
    t["mvir"], _, _ = reshape_branches(
        edges, snap, tree_file.read(["Group_M_TopHat200"])[0][t_ok])
    subfind_id = tree_file.read(["SubhaloIDRaw"])[0][t_ok] % 100000000000
    t["subfind_id"], _, _ = reshape_branches(edges, snap, subfind_id)
    t["vmax"], _, _ = reshape_branches(edges, snap, tree_file.read(["SubhaloVmax"])[0][t_ok])

    if is_hydro:
        t['stellar_mass'], _, _ = reshape_branches(
        edges, snap, tree_file.read(["SubhaloMassInRadType"])[0][t_ok, 4])

    sub_id = tree_file.read(["SubhaloID"])[0][t_ok]
    first_sub = tree_file.read(["FirstSubhaloInFOFGroupID"])[0][t_ok]
    is_sub = first_sub != sub_id
    t["is_sub"], _, _ = reshape_branches(edges, snap, is_sub)

    first_sub_idx = id_to_index(sub_id, first_sub)

    for snap in range(t.shape[1]):
        t["subfind_id"][~t["ok"][:,snap],snap] = -1

    for i in range(len(t)):
        t["first_sub_idx"][i,:] = tree_to_branch[t["first_sub_idx"][i,:]]

    # extract other global statistics about the branches such as 'mpeak_pre'
    b = process_branches(t, t["ok"])
    if read_matches:
        ok = (b["mpeak_pre"] > MIN_MASS) & (~b["is_err"])
    else:
        ok = (b["mpeak_pre"] > MIN_MASS*MIN_BUFFER) & (~b["is_err"])
    t, b = t[ok], b[ok]

    if read_matches:
        matches = rt.Matches(match_file).hydro_to_dmo
        for snap in range(100):
            t["match"][:,snap] = matches[snap][t["subfind_id"][:,snap]]
            t["match"][t["subfind_id"][:,snap] == -1,snap] = -1

    return t, b

def process_branches(t, ok):
    dtype = [("infall_snap", "i4"), ("mpeak", "f4"), ("mpeak_pre", "f4"),
             ("is_err", "?"), ("vpeak", "f4"), ("vpeak_pre", "f4")]
    b = np.zeros(len(t), dtype=dtype)

    for snap in range(t.shape[1]):
        b["mpeak"] = np.maximum(b["mpeak"], t["mdm"][:, snap])
        b["vpeak"] = np.maximum(b["vpeak"], t["vmax"][:, snap])

    snap = np.arange(100, dtype=int)

    for i in range(len(t)):
        mpeak_i = b["mpeak"][i]
        first_sub_mpeak = b["mpeak"][t["first_sub_idx"][i,:]]
        change_first_sub = (mpeak_i > first_sub_mpeak) & t["ok"][i,:]
        t["first_sub_idx"][i,change_first_sub] = i
        t["is_sub"][i,:] = (t["first_sub_idx"][i,:] != i) & t["is_sub"][i,:]

        if np.sum(t["is_sub"][i,:]) == 0:
            b["infall_snap"][i] = 100
        else:
            b["infall_snap"][i] = np.min(snap[t["is_sub"][i,:]])

        if b["infall_snap"][i] == 0:
            b["mpeak_pre"][i] = 0
            b["vpeak_pre"][i] = 0
        else:
            b["mpeak_pre"][i] = np.max(t["mdm"][i,:b["infall_snap"][i]])
            b["vpeak_pre"][i] = np.max(t["vmax"][i,:b["infall_snap"][i]])


    b["is_err"] = b["mpeak_pre"] == 0

    return b

class IDLookupTable(object):
    def __init__(self, ids):
        orig_idx = np.arange(len(ids), dtype=int)
        order = np.argsort(ids)
        self.s_ids = ids[order]
        self.s_idx = orig_idx[order]

    def find(self, ids):
        i = np.searchsorted(self.s_ids, ids)
        i = np.minimum(len(self.s_ids) - 1, i)
        i = np.maximum(0, i)
        ok = ids == self.s_ids[i]
        idx = self.s_idx[i]
        return idx, ok


def match_branches(t1, b1, t2, b2, post_infall=False):
    rt._ResetPairCounts()
    
    for snap in range(1, 100):
        match_1_to_2 = t1["match"][:,snap]
        is_err = match_1_to_2 == -1

        id_1, id_2 = t1["subfind_id"][:,snap], t2["subfind_id"][:,snap]
        table_1, table_2 = IDLookupTable(id_1), IDLookupTable(id_2)

        match_1 = np.arange(len(match_1_to_2), dtype=np.int64)

        bi_1 = match_1#, ok1 = table_1.find(match_1)
        ok1 = t1["ok"][:,snap]
        bi_1[~ok1] = -1
        bi_2, ok2 = table_2.find(match_1_to_2)
        ok = ok1 & ok2
        bi_1, bi_2, is_err = bi_1[ok], bi_2[ok], is_err[ok]
        if not post_infall:
            infall_snap_1 = b1["infall_snap"][bi_1]
            infall_snap_2 = b2["infall_snap"][bi_2]
            bi_1[infall_snap_1 <= snap] = -1
            bi_2[infall_snap_2 <= snap] = -1

        ok = (bi_1 != -1) & (bi_2 != -1)
        #print(bi_1[ok])
        #print(bi_2[ok])
        #print()
        rt._AddPairs(bi_1[ok], bi_2[ok], np.sum(ok))

    b_1_to_2 = np.zeros(len(t1), dtype=np.int64)
    rt._MatchPairs(b_1_to_2, len(b_1_to_2))
    return b_1_to_2

def is_iso(t, b, i):
    if b["infall_snap"][i] == -1:
        return np.ones(t.shape[1], dtype=bool)
    snap = np.arange(t.shape[1], dtype=int)
    return (snap < b["infall_snap"][i]) & t["ok"][i,:]


def main():
    import matplotlib.pyplot as plt
    try:
        import palette
        palette.configure(False)
    except:
        pass

    tree_file_bar = rt.Tree(dir_bar)
    # t_bar -> 2d array containing tree data. It is smaller than a full
    # sized tree.
    # b_bar -> array of "branches", where element annotates a tree branch
    t_bar, b_bar = read_small_reshaped_tree(tree_file_bar, is_hydro=True, read_matches=True)

    tree_file_dmo = rt.Tree(dir_dmo)
    t_dmo, b_dmo = read_small_reshaped_tree(tree_file_dmo, is_hydro=False)

    # b_1_to_2 gives, for each baryonic branch, the corresponding dmo branch
    b_1_to_2 = match_branches(t_bar, b_bar, t_dmo, b_dmo)

    print(t_bar.shape, t_dmo.shape)
    print(np.sum(b_1_to_2 == -1), len(b_1_to_2))

    fig, ax = plt.subplots()
    snap = np.arange(100)

    print(b_1_to_2[:20])

    # cache mahs and matching
    pickle.dump(t_dmo, open(CACHE_DIR + "/tng100_tdmo.pkl", 'wb'))
    pickle.dump(b_dmo, open(CACHE_DIR + "/tng100_bdmo.pkl", 'wb'))
    pickle.dump(t_bar, open(CACHE_DIR + "/tng100_tbar.pkl", 'wb'))
    pickle.dump(b_bar, open(CACHE_DIR + "/tng100_bbar.pkl", 'wb'))
    pickle.dump(b_1_to_2, open(CACHE_DIR + "/matches.pkl", 'wb'))


    for i1 in range(10):
        i2 = b_1_to_2[i1]
        if i2 == -1: continue

        ok1, ok2 = t_bar[i1]["ok"], t_dmo[i2]["ok"]

        ax.cla()

        ax.plot(snap[ok1], t_bar["mdm"][i1,ok1], c="tab:red")
        ax.plot(snap[ok1], t_bar["mdm"][i1,ok1], c="tab:red", ls="--")
        ax.plot(snap[ok2], t_dmo["mdm"][i2,ok2], c="tab:blue")
        ax.plot(snap[ok2], t_dmo["mdm"][i2,ok2], c="tab:blue", ls="--")

        ax.set_yscale("log")
        ax.set_xlabel(r"${\rm snap}$")
        ax.set_ylabel(r"$M\ (h^{-1}\,M_\odot)$")

        fig.savefig("plots/matches/match_%02d_tng100.png" % i1)

if __name__ == "__main__": main()
