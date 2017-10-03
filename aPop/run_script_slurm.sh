#!/bin/bash
################################################################################
#SBATCH --job-name Population_slurmtest
#SBATCH --time 48:00:00
#SBATCH -o population_output1.txt
#SBATCH -e population_output_error1.txt
#SBATCH --ntasks 160
#SBATCH --mem-per-cpu=4000MB
#SBATCH --mail-type=ALL

################################################################################
unset DISPLAY # slurm appear to create a problem with too many displays
srun --mpi=pmi2 python Population.py MPI