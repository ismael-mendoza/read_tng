import numpy as np
import h5py
import os
import os.path
import glob

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

        print(out)
        
        for file_name in self.file_names:
            f = h5py.File(file_name, "r")

            valid_names = list(f.keys())
            for i, var_name in enumerate(var_names):
                if var_name not in valid_names:
                    raise ValueError("variable '%s' not in file" % var_name)
                out[i].append(np.array(f[var_name]))

        print(out)

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
        
def main():
    sim = Haloes("TNG50-3-Dark")
    group_first_sub, group_n_sub = sim.read("Group", ["FirstSub", "Nsubs"])
    sub_group_num = sim.read("Subhalo", ["GrNr"])[0]
    for i in range(100):
        start, end = group_first_sub[i], group_first_sub[i] + group_n_sub[i]
        print(sub_group_num[start: end])
        
if __name__ == "__main__":
    main()
