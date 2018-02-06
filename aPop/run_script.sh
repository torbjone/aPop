#!/bin/bash
################################################################################
#SBATCH --job-name stick_population
#SBATCH --time 48:00:00
#SBATCH -o output_Ih2.txt
#SBATCH -e output_eIh2.txt
#SBATCH --ntasks 16
#SBATCH --mem-per-cpu=4000MB
#SBATCH --mail-type=ALL

################################################################################
unset DISPLAY # slurm appear to create a problem with too many displays
srun --mpi=pmi2 python Population.py MPI