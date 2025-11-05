#!/bin/bash
#SBATCH --job-name zoomin_7
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition sdicnormal

source ~/.bashrc
cd $PWD

# export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_PROC_BIND=spread
# export OMP_PLACES=cores

#julia -t 8 ES_mu.jl
julia -t 8 find_critical_point.jl

