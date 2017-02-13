#!/bin/bash

#####################################################
# example for a job where we consume lots of memory #
#####################################################

#SBATCH --job-name=test1

# we ask for 1 node
#SBATCH --nodes=1

# run for five minutes
#              d-hh:mm:ss
#SBATCH --time=0-00:05:00

# short partition should do it
#SBATCH --partition short

# total memory for this job
# this is a hard limit
# note that if you ask for more than one CPU has, your account gets
# charged for the other (idle) CPUs as well
#SBATCH --mem=32000MB
#SBATCH --mem-per-cpu=4GB

# turn on all mail notification
#SBATCH --mail-type=ALL

# you may not place bash commands before the last SBATCH directive
cd /global/work/torbness/aPop/aPop
python Population.py initialize
mpirun -np 8 python Population.py MPI
mpirun -np 8 python Population.py sum


# happy end
exit 0