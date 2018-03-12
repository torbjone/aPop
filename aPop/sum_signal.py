


import sys
from Population import sum_and_remove

sim_name = 'hay_classic'     # Control simulation
if sim_name == "stick_generic":
    from aPop.param_dicts import stick_population_params as param_dict
elif sim_name == "hay_generic":
    from aPop.param_dicts import generic_population_params as param_dict
elif sim_name == "hay_classic":
    from aPop.param_dicts import classic_population_params as param_dict
elif sim_name == "hbp_classic":
    from aPop.param_dicts import hbp_population_params as param_dict
elif sim_name == "generic_multimorph":
    from aPop.param_dicts import multimorph_population_params as param_dict
else:
    raise RuntimeError("Unrecognized conductance")

if "generic" in sim_name:
    param_dict.update({'input_region': sys.argv[1],
                   'correlation': float(sys.argv[2]),
                   'distribution': "linear_increase",
                   'conductance_type': 'generic',
                   'mu': None if sys.argv[3] == "None" else float(sys.argv[3]),
                   'holding_potential': -80.,
                   'cell_number': 0})
else:
    param_dict.update({'input_region': sys.argv[1],
                       'correlation': float(sys.argv[2]),
                       'conductance_type': 'Ih_plateau',
                       'holding_potential': -70.,
                       'cell_number': 0})
num_cells = int(sys.argv[-1])

sum_and_remove(param_dict, num_cells)