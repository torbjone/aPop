import sys
from aPop.param_dicts import classic_population_params as param_dict
from aPop.NeuralSimulation import NeuralSimulation

param_dict.update({'input_region': sys.argv[1],
                   'correlation': float(sys.argv[2]),
                   'conductance_type': 'Ih_plateau2',
                   'holding_potential': -70.,
                   'cell_number': int(sys.argv[3])})
ns = NeuralSimulation(**param_dict)
ns.run_single_simulation()