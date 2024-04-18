#!/bin/bash
#SBATCH --job-name=tng100_1
#SBATCH --output=%j.out
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=180GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=imendoza@umich.edu
#SBATCH --account=cavestru0
#SBATCH --partition=standard
module load python3.10-anaconda/2023.03
python3 small_match.py
