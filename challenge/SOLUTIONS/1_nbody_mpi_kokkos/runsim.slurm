#!/bin/bash

#SBATCH --job-name=nbody
#SBATCH --partition=kura-all
#SBATCH --output=%x-stdout_%j
#SBATCH --error=%x-stderr_%j
#SBATCH --time=0-00:20:00

#SBATCH --nodes=10

if [ $SLURM_JOB_PARTITION == 'kura-all' ]; then
  . /opt/Modules/3.2.10/init/sh
fi

module --force purge

module load gcc/11.1.0
module load mpich
module load vtk/9.4.1-mpi
module load cmake/3.26.0
module load kokkos/openmp

export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export KOKKOS_NUM_THREADS=2

srun -N 10 ./build/nbody_problem 2500 2000 100 0.001

