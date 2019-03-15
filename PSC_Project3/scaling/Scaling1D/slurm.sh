#!/bin/bash
#SBATCH --job-name=mpi_scaling
#SBATCH --output=procs64.o%j
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cluster=mpi
#SBATCH --partition=opa
#SBATCH --time=24:00:00

module purge
module load gcc/5.4.0
module load mpich/3.1

mpicc -o MPI.exe -O3 -lm mpiHeat_1v3.c
##mpirun -np 1 ./MPI.exe
mpirun -np $SLURM_NTASKS ./MPI.exe

