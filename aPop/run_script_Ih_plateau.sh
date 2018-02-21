#!/bin/bash
################################################################################
#SBATCH --job-name Ih
#SBATCH --time 10:00:00
#SBATCH -o output_Ih.txt
#SBATCH -e output_eIh.txt
#SBATCH --ntasks 80
#SBATCH --mem-per-cpu=2000MB
#SBATCH --mail-type=ALL

################################################################################
unset DISPLAY # slurm appear to create a problem with too many displays

# python Ih_plateau_simulations.py initialize
mpirun -np 80 python Ih_plateau_simulations.py MPI