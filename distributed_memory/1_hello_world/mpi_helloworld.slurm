#!/bin/bash
#SBATCH --job-name=helloworld_serial
#SBATCH --partition=nu
#SBATCH --output=stdout-%x_%j
#SBATCH --error=stderr-%x_%j
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1

module load gcc/11.1.0
module load mpich/4.1.1

CXX=mpic++

$CXX -o mpi_helloworld helloworld.cpp

mpirun -np 2 ./mpi_helloworld

rm mpi_helloworld
