#!/bin/bash
################################################################################
#SBATCH --job-name stick_population
#SBATCH --time 48:00:00
#SBATCH -o output_stick.txt
#SBATCH -e output_error_stick.txt
#SBATCH --ntasks 80
#SBATCH --mem-per-cpu=4000MB
#SBATCH --mail-type=ALL

################################################################################
unset DISPLAY # slurm appear to create a problem with too many displays
mpirun -np $SLURM_NTASKS python Population.py MPI
