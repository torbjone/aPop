
#!/bin/bash

#PBS -lnodes=1:ppn=1
#PBS -lwalltime=2:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

cd /global/work/torbness/aPop/population/simulations

rm center*_sig_generic_population_hay_*_generic_*_*_*_0???1.npy
rm center*_sig_generic_population_hay_*_generic_*_*_*_0???2.npy
rm center*_sig_generic_population_hay_*_generic_*_*_*_0???3.npy
rm center*_sig_generic_population_hay_*_generic_*_*_*_0???4.npy
rm center*_sig_generic_population_hay_*_generic_*_*_*_0???5.npy
rm center*_sig_generic_population_hay_*_generic_*_*_*_0???6.npy
rm center*_sig_generic_population_hay_*_generic_*_*_*_0???7.npy
rm center*_sig_generic_population_hay_*_generic_*_*_*_0???8.npy
rm center*_sig_generic_population_hay_*_generic_*_*_*_0???9.npy

rm lateral*_sig_generic_population_hay_*_generic_*_*_*_0???1.npy
rm lateral*_sig_generic_population_hay_*_generic_*_*_*_0???2.npy
rm lateral*_sig_generic_population_hay_*_generic_*_*_*_0???3.npy
rm lateral*_sig_generic_population_hay_*_generic_*_*_*_0???4.npy
rm lateral*_sig_generic_population_hay_*_generic_*_*_*_0???5.npy
rm lateral*_sig_generic_population_hay_*_generic_*_*_*_0???6.npy
rm lateral*_sig_generic_population_hay_*_generic_*_*_*_0???7.npy
rm lateral*_sig_generic_population_hay_*_generic_*_*_*_0???8.npy
rm lateral*_sig_generic_population_hay_*_generic_*_*_*_0???9.npy
