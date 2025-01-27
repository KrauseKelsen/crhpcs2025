#!/bin/sh

#SBATCH --job-name=program
#SBATCH --partition=nu
#SBATCH --output=stdout-%x_%j
#SBATCH --error=stderr-%x_%j
#SBATCH --time=0-00:20:00

#SBATCH --nodes=1

module purge
# Load needed modules

module load cuda/12.4.0
module load gcc/11.1.0
module load cmake/3.26.0
module load kokkos/cuda

<program> <arguments>
