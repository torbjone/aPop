import os
import sys
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False

from aPop import Population
from aPop.param_dicts import ih_plateau_population_params as param_dict

if sys.argv[1] == 'initialize':
    Population.initialize_population(param_dict)
    Population.plot_population(param_dict)
elif sys.argv[1] == 'MPI':
    Population.PopulationMPIclassicFork(param_dict)
