
#!/bin/bash

#PBS -lnodes=1:ppn=1
#PBS -lwalltime=10:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

cd /global/work/torbness/aPop/aPop
python Figures.py