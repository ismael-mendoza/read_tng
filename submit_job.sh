#!/bin/bash
#SBATCH --job-name=tng100-test1-07-13-2025
#SBATCH --output=jobs/%j.out
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=150GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=imendoza@umich.edu
#SBATCH --account=cavestru0
#SBATCH --partition=standard
module load python3.10-anaconda/2023.03
python3 small_match.py
# python3 small_match_tng50_3.py
