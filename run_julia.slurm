#!/bin/bash
#SBATCH --job-name Chiral_PEPS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition sdicnormal

source ~/.bashrc
cd $PWD
julia -t 16 ES_mu.jl