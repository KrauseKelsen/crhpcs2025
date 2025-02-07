#!/bin/bash
#SBATCH --job-name=findpi_mpi
#SBATCH --partition=nu
#SBATCH --output=stdout-%x_%j
#SBATCH --error=stderr-%x_%j
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1
#SBATCH --nodelist=nu-1c.cnca

module load gcc/11.1.0
module load mpich/4.1.1 

CXX=mpic++

$CXX -o findpi_mpi findpi_mpi.cpp

mpirun -np 4 ./findpi_mpi 1000

rm findpi_mpi
