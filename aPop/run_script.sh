#!/bin/bash

#PBS -lnodes=8:ppn=16
#PBS -lwalltime=60:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

cd /global/work/torbness/aPop/aPop
#python Population.py initialize
mpirun -np 128 python Population.py MPI
#mpirun -np 64 python Population.py sum
