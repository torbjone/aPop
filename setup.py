#!/usr/bin/env python
import os
from os.path import join
import shutil
from distutils.core import setup

#  try and locate the nrnivmodl script of NEURON in PATH so that the
#  NEURON *.mod files can be compiled in place
from distutils.spawn import find_executable, spawn

cwd = os.getcwd()

if find_executable('nrnivmodl') is not None:

    for mod_path in [join("aPop", "neuron_models"),
                     join("aPop", "neuron_models", 'hay', 'mod'),
                     join("aPop", "neuron_models", 'hbp_cells', 'mods')]:
        os.chdir(join(mod_path))
        for path in ['x86_64', 'i686', 'powerpc']:
            if os.path.isdir(path):
                shutil.rmtree(path)
        spawn([find_executable('nrnivmodl')])
        os.chdir(cwd)
else:
    print("nrnivmodl script not found in PATH, thus NEURON .mod files could" +
          "not be compiled")


setup(name='aPop',
      version='1.0',
      description='Effect of Active conductances on the LFP',
      author='Torbjorn V. Ness',
      author_email='torbness@gmail.com',
      url='https://github.com/torbjone/aPop.git',
      packages=['aPop'],
      package_data={'aPop': [join('neuron_models', "*.mod"),
                             join('neuron_models', "x86_64", "*"),
                             join('neuron_models', "i686", "*"),
                             join('neuron_models', "powerpc", "*"),

                             join('neuron_models', 'hay', "mod", "x86_64", "*"),
                             join('neuron_models', 'hay', "mod", "i686", "*"),
                             join('neuron_models', 'hay', "mod", "powerpc", "*"),
                             join('neuron_models', 'hay', "*.hoc"),
                             join('neuron_models', 'hay', "*.rot"),
                             join('neuron_models', 'hay', "*.py"),

                             join('neuron_models', 'stick_cell', "*.py"),
                             join('neuron_models', 'stick_cell', "*.hoc"),

                             join('neuron_models', 'neuron_nmo_adult', "*.hoc"),
                             join('neuron_models', 'neuron_nmo_adult', "*.py"),
                             join('neuron_models', 'neuron_nmo_adult', "*", "CNG version", "*.swc"),

                             join('neuron_models', 'hbp_cells', "mods", "x86_64", "*"),
                             join('neuron_models', 'hbp_cells', "mods", "i686", "*"),
                             join('neuron_models', 'hbp_cells', "mods", "powerpc", "*"),
                             join('neuron_models', 'hbp_cells', "*.py"),
                             join('neuron_models', 'hbp_cells', "L*", "*.py"),
                             join('neuron_models', 'hbp_cells', "L*", "*.hoc"),
                             join('neuron_models', 'hbp_cells', "L*", "morphology", "*"),
                             join('neuron_models', 'hbp_cells', "L*", "synapses", "*.hoc"),
                             ]}
     )