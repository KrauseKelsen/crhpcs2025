#!/bin/bash
#SBATCH --job-name=helloworld_parallel
#SBATCH --partition=nu
#SBATCH --output=stdout-%x_%j
#SBATCH --error=stderr-%x_%j
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1
#SBATCH --nodelist=nu-1c.cnca

module load gcc/11.1.0
module load mpich/4.1.1

CXX=mpic++

$CXX -o mpi_helloworld helloworld.cpp

mpirun -np 4 ./mpi_helloworld

rm mpi_helloworld
