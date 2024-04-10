import numpy as np
import h5py
import os
import os.path
import glob
import ctypes
import numpy.ctypeslib as ctypeslib
import time
import matplotlib.pyplot as plt

#Annotate tpyes of the Go library
read_tng_go = ctypes.cdll.LoadLibrary("/home/users/phil1/code/src/github.com/phil-mansfield/read_tng/read_tng.so")

_ResetPairCounts = read_tng_go.ResetPairCounts
_ResetPairCounts.restype = None
_ResetPairCounts.argtypes = [
]

_IDToIndex = read_tng_go.IDToIndex
_IDToIndex.restype = None
_IDToIndex.argtypes = [
    ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
    ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
    ctypes.c_longlong,
    ctypes.c_longlong,
]

_AddPairs = read_tng_go.AddPairs
_AddPairs.restype = None
_AddPairs.argtypes = [
    ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
    ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
    ctypes.c_longlong,
]

_MatchPairs = read_tng_go.MatchPairs
_MatchPairs.restype = None
_MatchPairs.argtypes = [
    ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
    ctypes.c_longlong,
]

def list_hdf5_files(dir):
    """ list_hdf5_files returns all the hdf5 files in a direcotry.
    """
    if type(dir) == list:
        return dir
    elif os.path.isfile(dir):
        return [dir]

    pattern = os.path.join(dir, "*.hdf5")
    file_names = np.array(glob.glob(pattern))
    idx = np.array([int(fname.split(".")[-2]) for fname in file_names])
    return file_names[np.argsort(idx)]
    

def _flatten(arrays):
    """ _flatten takes a list of numpy arrays and flattens it into a single
    array. arrays[i].shape[0] can be any value, but all other components of the
    shape vectors must be the same.
    """

    N = sum(arr.shape[0] for arr in arrays)

    shape, dtype = arrays[0].shape, arrays[0].dtype
    if len(shape) == 1:
        out = np.zeros(N, dtype=dtype)
    else:
        out = np.zeros((N,) + shape[1:], dtype=dtype)

    start, end = 0, 0
    for i in range(len(arrays)):
        end += arrays[i].shape[0]
        out[start: end] = arrays[i]
        start = end

    return out

class Haloes(object):
    def __init__(self, dir):
        """ Haloes creates a Haloes object with the hdf5 files in 
        a given directory. Haloes has a few properties specifying
        cosmological and simulation parameters:

        h100    - H0(z=0) / (100 km/s/Mpc)
        omega_m - Omega_m(z=0)
        L       - Box size on one size (comoving Mpc/h)
        z       - Redshift
        scale   - Scale factor, 1/(1+z)
        """
        self.file_names = list_hdf5_files(dir)
        
        f = h5py.File(self.file_names[0], "r")
        
        self.h100 = f["Header"].attrs["HubbleParam"]
        self.omega_m = f["Header"].attrs["Omega0"]
        self.L = f["Header"].attrs["BoxSize"] / 1e3
        self.z = f["Header"].attrs["Redshift"]
        self.scale = f["Header"].attrs["Time"]

    def read(self, group_name, var_names):
        """ read reads serveral variables from the simulation and returns them
        as a list of arrays. The group_name is either "Group" or "Subhalo" and
        var_names is a list of variable names that you want to read. The
        variables available for FoF groups are given here:
        https://www.tng-project.org/data/docs/specifications/#sec2a
        For subhaloes:
        https://www.tng-project.org/data/docs/specifications/#sec2b

        For example, to read "GroupPos", "GroupVel", "Group_M_Crit200", you
        would set group_name to "Group" and var_names to ["Pos", "Vel",
        "M_Crit200"].

        Results are converted out of TNG's inhomogenous units. Instead all
        positions and distances are in units of comoving Mpc/h, velocities
        are physical km/s, and masses are Msun/h.
        """
        out = [[] for _ in var_names]

        for file_name in self.file_names:
            f = h5py.File(file_name, "r")
            if len(f[group_name].keys()) == 0: continue

            for i, var_name in enumerate(var_names):
                if "_" in var_name: var_name = "_" + var_name
                
                data_set = f[group_name]["%s%s" % (group_name, var_name)]
                out[i].append(np.array(data_set))


        for i, var_name in enumerate(var_names):
            out[i] = _flatten(out[i])
            out[i] = self._convert_units(var_name, out[i])

        return out

    def _convert_units(self, var_name, x):
        """ _convert_units(var_name, x) converts the array, x, with the variable
        name, var_name, from TNG's inhomogenous units to Simulation's
        standardized units.
        """
        if "Mass" in var_name or "M_" in var_name:
            return x * 1e10
        elif var_name == "Pos" or "R_" in var_name or "Rad" in var_name:
            return x / 1e3
        else:
            return x

class Tree(object):
    def __init__(self, dir):
        """ Tree creates Tree object with the hdf5 files in 
        a given directory.
        """
        self.file_names = list_hdf5_files(dir)

    def read(self, var_names):
        """ read reads serveral variables from the simulation and returns them
        as a list of arrays.

        Results are converted out of TNG's inhomogenous units. Instead all
        positions and distances are in units of comoving Mpc/h, velocities
        are physical km/s, and masses are Msun/h.
        """
        out = [[] for _ in var_names]
        

        for file_name in self.file_names:
            f = h5py.File(file_name, "r")

            valid_names = list(f.keys())
            for i, var_name in enumerate(var_names):
                if var_name not in valid_names:
                    raise ValueError("variable '%s' not in file" % var_name)
                out[i].append(np.array(f[var_name]))

        for i, var_name in enumerate(var_names):
            out[i] = _flatten(out[i])
            out[i] = self._convert_units(var_name, out[i])

        return out

    def _convert_units(self, var_name, x):
        """ _convert_units(var_name, x) converts the array, x, with the variable
        name, var_name, from TNG's inhomogenous units to Simulation's
        standardized units.
        """
        if "Mass" in var_name or "M_" in var_name:
            return x * 1e10
        elif var_name == "Pos" or "R_" in var_name or "Rad" in var_name:
            return x / 1e3
        else:
            return x

class Matches(object):
    def __init__(self, name):
        f = h5py.File(name, "r")
        self.hydro_to_dmo = [
            #np.array(f["Snapshot_%d/SubhaloIndexDark_SubLink" % i])
            np.array(f["Snapshot_%d/SubhaloIndexDark_LHaloTree" % i])
            for i in range(100)
        ]
        
def branch_edges(first_prog):
    mids = np.where(first_prog == -1)[0] + 1
    edges = np.zeros(len(mids) + 2, dtype=int)
    edges[1:-1] = mids
    edges[-1] = len(first_prog)
    return edges
    
def reshape_branches(edges, snap, x):
    start, end = edges[:-1], edges[1:]
    n_branch, n_snap = len(start), 100

    xx = np.zeros((n_branch, n_snap), dtype=x.dtype)
    ok = np.zeros((n_branch, n_snap), dtype=bool)
    tree_to_branch = np.zeros(len(x), dtype=np.int64)

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

def read_reshaped_tree(tree_file):
    (snap, sub_id, first_sub,
     first_prog, subfind_id_raw,
     rvir, mvir, group_x,
     vmax, x,
     msub_type, r_half_type) = tree_file.read(
         ["SnapNum", "SubhaloID", "FirstSubhaloInFOFGroupID",
          "FirstProgenitorID", "SubhaloIDRaw",
          "Group_R_TopHat200", "Group_M_TopHat200", "GroupPos",
          "SubhaloVmax", "SubhaloPos",
          "SubhaloMassType", "SubhaloHalfmassRadType"]
    )

    m_dm, m_star = msub_type[:,1], msub_type[:,3]
    r_half_dm, r_half_star = msub_type[:,1], msub_type[:,3]
    is_sub = sub_id != first_sub
    subfind_id = subfind_id_raw % 100000000000

    first_sub_idx = id_to_index(sub_id, first_sub)

    dtype = [("rvir", "f4"), ("mvir", "f4"), ("group_x", "f4", 3),
             ("vmax", "f4"), ("x", "f4", 3),
             ("mdm", "f4"), ("mstar", "f4"),
             ("rhalf_dm", "f4"), ("rhalf_star", "f4"),
             ("subfind_id", "i8"),
             ("is_sub", "?"), ("first_sub_idx", "i8"), ("ok", "?")]
    t = np.zeros(len(rvir), dtype=dtype)
    t["rvir"], t["mvir"], t["group_x"] = rvir, mvir, group_x
    t["vmax"], t["x"] = vmax, x
    t["mdm"], t["mstar"] = m_dm, m_star
    t["rhalf_dm"], t["rhalf_star"] = r_half_dm, r_half_star
    t["subfind_id"] = subfind_id
    t["is_sub"], t["first_sub_idx"] = is_sub, first_sub_idx

    edges = branch_edges(first_prog)
    t, ok, tree_to_branch = reshape_branches(edges, snap, t)

    t["ok"] = ok
    for snap in range(t.shape[1]):
        t["subfind_id"][~t["ok"][:,snap],snap] = -1

    for i in range(len(t)):
        t["first_sub_idx"][i,:] = tree_to_branch[t["first_sub_idx"][i,:]]

    b = process_branches(t, ok)
    return t, b

def process_branches(t, ok):
    dtype = [("infall_snap", "i4"), ("mpeak", "f4"), ("mpeak_pre", "f4"),
             ("is_err", "?")]
    b = np.zeros(len(t), dtype=dtype)

    for snap in range(t.shape[1]):
        b["mpeak"] = np.maximum(b["mpeak"], t["mdm"][:,snap])

    snap = np.arange(100, dtype=int)

    for i in range(len(t)):
        if i % 100000 == 0:
            print(i)

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
        else:
            b["mpeak_pre"][i] = np.max(t["mdm"][i,:b["infall_snap"][i]])


    b["is_err"] = b["mpeak_pre"] == 0

    return b

def match_branches(t1, b1, t2, b2, matches, post_infall=False):
    _ResetPairCounts()
    
    n_id_1, n_id_2 = len(t1), len(t2)
    tot_halo, tot_match = 0, 0
    for snap in range(1, 100):
        if snap % 20 == 0: print(snap)
        match_1_to_2 = matches.hydro_to_dmo[snap]

        tot_halo += len(match_1_to_2)
        tot_match += np.sum(match_1_to_2 >= 0)

        n_idx_1 = np.max(t1["subfind_id"][:,snap]) + 1
        n_idx_2 = np.max(t2["subfind_id"][:,snap]) + 1
        idx_1 = -1*np.ones(n_idx_1, dtype=np.int64)
        idx_2 = -1*np.ones(n_idx_2, dtype=np.int64)

        subfind_1 = np.ascontiguousarray(t1["subfind_id"][:,snap])
        subfind_2 = np.ascontiguousarray(t2["subfind_id"][:,snap])
        _IDToIndex(subfind_1, idx_1, n_id_1, n_idx_1)
        _IDToIndex(subfind_2, idx_2, n_id_2, n_idx_2)

        match_1 = np.arange(len(match_1_to_2), dtype=np.int64)
        match_1[match_1_to_2 == -1] = -1

        bi_1 = idx_1[match_1]
        bi_2 = idx_2[match_1_to_2]
        if not post_infall:
            infall_snap_1 = b1["infall_snap"][bi_1]
            infall_snap_2 = b2["infall_snap"][bi_2]
            bi_1[infall_snap_1 <= snap] = -1
            bi_2[infall_snap_2 <= snap] = -1

        is_err = match_1_to_2 == -1
        bi_1[is_err], bi_2[is_err] = -1, -1

        _AddPairs(bi_1, bi_2, len(bi_1))

    b_1_to_2 = np.zeros(len(t1), dtype=np.int64)
    _MatchPairs(b_1_to_2, len(b_1_to_2))
    return b_1_to_2

def is_iso(t, b, i):
    if b["infall_snap"][i] == -1:
        return np.ones(t.shape[1], dtype=bool)
    snap = np.arange(t.shape[1], dtype=int)
    return (snap < b["infall_snap"][i]) & t["ok"][i,:]

def main():
    try:
        import palette
        from palette import pc
        palette.configure(False)
    except:
        pc = lambda x: x

    print("Reading matches")
    halo_matches = Matches("/oak/stanford/orgs/kipac/users/phil1/simulations/TNG50_3/matches/subhalo_matching_to_dark.hdf5")

    print("Reading DMO")
    tree_file_dmo = Tree("/oak/stanford/orgs/kipac/users/phil1/simulations/TNG50_3_Dark/trees/sublink")
    t_dmo, b_dmo = read_reshaped_tree(tree_file_dmo)

    print("Reading baryons")
    tree_file_bar = Tree("/oak/stanford/orgs/kipac/users/phil1/simulations/TNG50_3/trees/sublink")
    t_bar, b_bar = read_reshaped_tree(tree_file_bar)

    matches = match_branches(t_bar, b_bar, t_dmo, b_dmo, halo_matches)

    cuts = [3e10, 3e11, 3e12, 3e13]
    for i_cut in range(len(cuts)):
        cut = cuts[i_cut]
        ok_1 = b_bar["mpeak_pre"] > cut
        ok_2 = ok_1 & (matches != -1)
        print("N", np.sum(ok_1))
        print("f_match", np.sum(ok_2)/np.sum(ok_1))
        print("Mhydro/Mdmo",
              np.median(b_bar["mpeak_pre"][ok_2]/b_dmo["mpeak_pre"][matches[ok_2]]))

    host_bar = np.argmax(b_bar["mpeak"])
    host_dmo = matches[host_bar]

    targets_bar = np.where(np.any(
        t_bar["first_sub_idx"] == host_bar, axis=1))[0]
    targets_dmo = matches[targets_bar]
    ok = targets_dmo != -1
    targets_bar, targets_dmo = targets_bar[ok], targets_dmo[ok]
    
    order = np.argsort(b_bar["mpeak"][targets_bar])[::-1]
    targets_bar, targets_dmo = targets_bar[order], targets_dmo[order]

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    x_ax, m_ax = axs[0], axs[1]

    snap = np.arange(100, dtype=int)

    dx_dmo = t_dmo["x"] - t_dmo["x"][host_dmo]
    dx_bar = t_bar["x"] - t_bar["x"][host_bar]

    for i in range(250):
        if i % 20 == 0: print(i)
        x_ax.cla()
        m_ax.cla()
        m_ax.set_xlabel(r"${\rm Snap}$")
        m_ax.set_ylabel(r"$\log_{10}(M_{\rm sub})$")
        x_ax.set_xlabel(r"$x\ (h^{-1}{\rm cMpc})$")
        x_ax.set_ylabel(r"$x\ (h^{-1}{\rm cMpc})$")

        i_bar, i_dmo = targets_bar[i], targets_dmo[i]
        x_ax.set_title(r"$M_{\rm peak} = %.2f,\ M_{\rm peak,pre} = %.2f$" % 
                       (np.log10(b_bar["mpeak"][i_bar]),
                        np.log10(b_bar["mpeak_pre"][i_bar])))

        ok_bar = t_bar["ok"][i_bar,:] & t_bar["ok"][host_bar,:]
        ok_dmo = t_dmo["ok"][i_dmo,:] & t_dmo["ok"][host_dmo,:]
        is_iso_bar = is_iso(t_bar, b_bar, i_bar) & ok_bar
        is_iso_dmo = is_iso(t_dmo, b_dmo, i_dmo) & ok_dmo

        m_ax.plot(snap[ok_dmo], np.log10(t_dmo["mdm"][i_dmo,ok_dmo]),
                  c="k", lw=1.5)
        m_ax.plot(snap[ok_bar], np.log10(t_bar["mdm"][i_bar,ok_bar]),
                  c=pc("r"), lw=1.5)
        m_ax.plot(snap[is_iso_dmo], np.log10(t_dmo["mdm"][i_dmo,is_iso_dmo]),
                  c="k")
        m_ax.plot(snap[is_iso_bar], np.log10(t_bar["mdm"][i_bar,is_iso_bar]),
                  c=pc("r"))

        x_ax.plot(dx_dmo[i_dmo,ok_dmo,0],
                  dx_dmo[i_dmo,ok_dmo,1],
                  c="k", lw=1.5)
        x_ax.plot(dx_bar[i_bar,ok_bar,0],
                  dx_bar[i_bar,ok_bar,1],
                  c=pc("r"), lw=1.5)
        x_ax.plot(dx_dmo[i_dmo,is_iso_dmo,0],
                  dx_dmo[i_dmo,is_iso_dmo,1],
                  c="k")
        x_ax.plot(dx_bar[i_bar,is_iso_bar,0],
                  dx_bar[i_bar,is_iso_bar,1],
                  c=pc("r"))
        
        max_dx = np.max(np.abs(dx_bar[i_bar,ok_bar,0]))
        max_dy = np.max(np.abs(dx_bar[i_bar,ok_bar,1]))
        max_dx = max(max_dx, max_dy)
        if max_dx != 0:
            x_ax.set_xlim(-max_dx, max_dx)
            x_ax.set_ylim(-max_dy, max_dy)

        fig.savefig("plots/comparisons/comp_%03d.png" % i)


if __name__ == "__main__":
    main()
