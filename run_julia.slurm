#!/bin/bash
#SBATCH --job-name ES_mu_5
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition sdicnormal

source ~/.bashrc
cd $PWD
julia -t 16 ES_mu.jl