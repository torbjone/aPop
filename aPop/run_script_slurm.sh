#!/bin/bash
################################################################################
#SBATCH --job-name Population_slurmtest
#SBATCH --time 200:00:00
#SBATCH -o population_output.txt
#SBATCH -e population_output_error.txt
#SBATCH --ntasks 160
#SBATCH --mem-per-cpu=2000MB
#SBATCH --mail-type=ALL

################################################################################
unset DISPLAY # slurm appear to create a problem with too many displays
mpirun -np $SLURM_NTASKS python Population.py MPI
