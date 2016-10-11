
#!/bin/bash

#PBS -lnodes=1:ppn=1
#PBS -lwalltime=1:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

cd /global/work/torbness/aPop/population/simulations

rm *sig_*classic*_0???0.npy
rm *sig_*classic*_0???2.npy
rm *sig_*classic*_0???3.npy
rm *sig_*classic*_0???4.npy
rm *sig_*classic*_0???5.npy
rm *sig_*classic*_0???6.npy
rm *sig_*classic*_0???7.npy
rm *sig_*classic*_0???8.npy
rm *sig_*classic*_0???9.npy
