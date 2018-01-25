# README #

The python code to reproduce all figures from a modelling study of the contribution
of active conductances to the Local Field Potential for cortical populations.
Figure numbers in Figures.py corresponds to figures in the paper.

For a related single cell study, see
https://www.ncbi.nlm.nih.gov/pubmed/27079755

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
mpirun -np 4 python Population.py
however to do all simulations needed to reproduce all the figures, a
supercomputer is recommended/needed. For any code-spesific questions or
concerns, please contact Torbjørn V Ness: torbness@gmail.com

### Who do I talk to? ###

Torbjørn V Ness - torbness@gmail.com

Michiel Remme

Gaute T Einevol - gaute.einevoll@nmbu.no
