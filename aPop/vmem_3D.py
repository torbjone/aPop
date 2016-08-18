from __future__ import division
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False
from plotting_convention import *
import pylab as plt
import numpy as np
import os
import sys
from os.path import join
from suppress_print import suppress_stdout_stderr
with suppress_stdout_stderr():
    import LFPy
from NeuralSimulation import NeuralSimulation
import tools
import scipy.fftpack as sf

from Population import PopulationMPIgeneric, initialize_population, plot_population, PopulationMPIclassic


if __name__ == '__main__':
    from param_dicts import vmem_3D_population_params as param_dict


    conductance = 'generic'
    if len(sys.argv) >= 3:
        cell_number = int(sys.argv[3])
        input_region = sys.argv[1]
        correlation = float(sys.argv[2])
        if conductance == 'generic' or conductance == 'stick_generic':
            distribution = sys.argv[4]
            mu = float(sys.argv[5])
            param_dict.update({'input_region': input_region,
                               'cell_number': cell_number,
                               'mu': mu,
                               'distribution': distribution,
                               'correlation': correlation,
                               })
        else:
            conductance_type = sys.argv[4]
            holding_potential = int(sys.argv[5])
            param_dict.update({'input_region': input_region,
                               'cell_number': cell_number,
                               'conductance_type': conductance_type,
                               'holding_potential': holding_potential,
                               'correlation': correlation,
                               })
        ns = NeuralSimulation(**param_dict)
        ns.run_distributed_synaptic_simulation()
    elif len(sys.argv) == 2 and sys.argv[1] == 'initialize':
        initialize_population(param_dict)
        plot_population(param_dict)
    elif len(sys.argv) == 2 and sys.argv[1] == 'MPI':
        if conductance == 'generic' or conductance == 'stick_generic':
            PopulationMPIgeneric()
        else:
            PopulationMPIclassic()
