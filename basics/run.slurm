#!/bin/sh

#SBATCH --job-name=program
#SBATCH --partition=nukwa-v100
#SBATCH --output=stdout-%x_%j
#SBATCH --error=stderr-%x_%j
#SBATCH --time=0-00:20:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

module purge
source /home/casch/.bashrc
# Load needed modules

module load cuda/12.4.0
module load gcc/11.1.0
module load cmake/3.26.0
module load kokkos/cuda

<program> <arguments>
