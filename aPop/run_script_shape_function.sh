#!/bin/bash

#PBS -lnodes=2:ppn=16
#PBS -lwalltime=60:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

cd /global/work/torbness/aPop/aPop
mpirun -np 32 python ShapeFunction.py MPIgeneric
python ShapeFunction.py sum

