# README #

The python code to reproduce all figures from a modelling study of the contribution
of active conductances to the Local Field Potential for cortical populations.
Figure numbers in Figures.py corresponds to figures in the paper.

For a related single cell study, see
https://www.ncbi.nlm.nih.gov/pubmed/27079755

with corresponding python repository: https://github.com/torbjone/aLFP

### Installation ###
The code should be installed in place. 

git clone https://github.com/torbjone/aPop.git

pip install -e aPop

or through 

git clone https://github.com/torbjone/aPop.git
cd aPop/
python setup.py build_ext --inplace

The code can also be used without installation provided the 
NEURON mod files have been compiled (write nrnivmodl in the terminal 
for all folders that contain *.mod files). 




### How do I get set up? ###
In all folders containing .mod files the command "nrnivmodl" 
(Linux and Mac) must be executed in a terminal. This assumes
that NEURON (www.neuron.yale.edu) is set up on the system and 
functioning properly. LFPy must also be installed.
This can be done by pip install, "pip install LFPy", but see

lfpy.github.io/information.html#installing-neuron-with-python

for more information on how to make NEURON and Python work together.

No attempts has been made for this to work at other operating systems 
than Linux, but we are happy to help people get started.

All simulations for a given cell model type are run with

*python Population.py initialize*

*mpirun -np 4 python Population.py MPI*

where the last line uses 4 processes. To run full populations,
it is strongly recommended to use a supercomputer with a large number of 
cores available. An example run script is included in run_script.sh

For any code-spesific questions or
concerns, please contact Torbjørn V Ness: torbness@gmail.com

### Simple example ###
The main findings of the study can be reproduced in a small simplified 
population model of stick neurons by opening a terminal in 
the aPop/aPop folder, and writing:

*python stick_pop_example.py*

or

*jupyter notebook index.ipynb*


### Cell models ###
The simulations rely on several different cell models that are 
available from NeuroMorpho.org, ModelDB, or the 
Blue Brain Project (bbp.epfl.ch/nmc-portal/). 
The cell models are also included in this repository for convenience.

### Code structure ###
NeuralSimulation.py : Responsible for all single-cell simulations

Population.py : Initialize and handle population simulations

Figures.py : Reproduce all result figures of the project after 
all simulations are done



### Who do I talk to? ###

Torbjørn V Ness - torbness@gmail.com

Michiel Remme

Gaute T Einevol - gaute.einevoll@nmbu.no
