#!/bin/bash
################################################################################
#SBATCH --job-name Ih_plat
#SBATCH --time 10:00:00
#SBATCH -o output_Ih2.txt
#SBATCH -e output_eIh2.txt
#SBATCH --ntasks 80
#SBATCH --mem-per-cpu=4000MB
#SBATCH --mail-type=ALL

################################################################################
unset DISPLAY # slurm appear to create a problem with too many displays

python Ih_plateau_simulations.py initialize
srun --mpi=pmi2 python Ih_plateau_simulations.py MPI