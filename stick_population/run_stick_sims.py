import sys
from aPop.param_dicts import stick_population_params as param_dict
from aPop.NeuralSimulation import NeuralSimulation
from aPop.Population import initialize_population, plot_population, PopulationMPIgeneric

sim_name = 'stick_generic'  # Figure 8CD

param_dict['save_folder'] = '.'

initialize_population(param_dict)
plot_population(param_dict)

PopulationMPIgeneric(param_dict)
