# read_tng
A simple Python library for reading Illustris-TNG catalogs and trees

## Build instructions

1. Install a Go compiler
2. If you're on x86, run `./build_script.sh` in this directory. This should generate a bunch of C file garbage.
3. Go to read_tng.py and edit line 12 to point to the absolute locaiton of the .so file that this generated.
