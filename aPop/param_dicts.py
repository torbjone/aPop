# -*- coding: utf-8 -*-
import os
from os.path import join
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False
import numpy as np


root_folder = os.path.split(os.path.split(__file__)[0])[0]

center_elec_z = np.arange(-200, 1201, 200)
center_elec_x = np.zeros(len(center_elec_z))
center_elec_y = np.zeros(len(center_elec_z))
center_elec_params = {
        'sigma': 0.3,
        'x': center_elec_x,
        'y': center_elec_y,
        'z': center_elec_z,
        'method': "soma_as_point",
}

stick_center_electrode_parameters = center_elec_params.copy()
stick_center_electrode_parameters['method'] = 'linesource'

# Time resolution of 2**-4 is almost identical to 2**-5
dt = 2**-4
end_T = 2**13 - dt
# end_T = 2**13 - dt
cut_off = 2000 #if at_stallo else 200




scale = 10
num_cells = 100 * scale ** 2
population_radius = 100. * scale
dr = 50.
population_radii = np.arange(dr, population_radius + dr, dr)
layer_5_thickness = 200  # From Markram (2015): Thickness L1-L5: 1382 um. Hay cell height: 1169 um. 1382 - 1169 = 213 um
generic_population_params = {'input_type': 'distributed_delta',
                             'name': 'generic_population',
                             'dt': dt,
                             'cell_name': 'hay',
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': layer_5_thickness,
                             'cut_off': cut_off,
                             'end_t': end_T,
                             'syn_tau': dt * 3,
                             'syn_weight': 0.03,
                             'max_freq': 500,
                             'holding_potential': -80,
                             'conductance_type': 'generic',
                             'save_folder': 'population',
                             'center_electrode_parameters': center_elec_params,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': [
                                               'homogeneous',
                                               "distal_tuft",
                                               "basal"
                                               ],
                             'mus': [-0.5, 0.0, 2.0, None],
                             'distributions': ['uniform',
                                               'linear_increase',
                                               'linear_decrease'],
                             'correlations': [0.0, 0.01, 0.1, 1.0]
                             }


ih_plateau_population_params = {'input_type': 'distributed_delta',
                             'name': 'classic_population',
                             'cell_name': 'hay',
                             'dt': dt,
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': layer_5_thickness,
                             'cut_off': cut_off,
                             'end_t': end_T,
                             'syn_tau': dt * 3,
                             'syn_weight': 0.03,
                             'max_freq': 500,
                             'holding_potential': -70,
                             'conductance_type': 'classic',
                             'save_folder': 'Ih_plateau',
                             # 'lateral_electrode_parameters': lateral_electrode_parameters,
                             'center_electrode_parameters': center_elec_params,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': ["homogeneous", "distal_tuft"],
                             'mus': None,
                             'holding_potentials': [-70],
                             'distributions': None,
                             'conductance_types': ['Ih'],
                             'correlations': [0.0, 1.0]
                             }

classic_population_params = {'input_type': 'distributed_delta',
                             'name': 'classic_population',
                             'cell_name': 'hay',
                             'dt': dt,
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': layer_5_thickness,
                             'cut_off': cut_off,
                             'end_t': end_T,
                             'syn_tau': dt * 3,
                             'syn_weight': 0.03,
                             'max_freq': 500,
                             'holding_potential': -70,
                             'conductance_type': 'classic',
                             'save_folder': 'population',
                             # 'lateral_electrode_parameters': lateral_electrode_parameters,
                             'center_electrode_parameters': center_elec_params,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': ["homogeneous", "distal_tuft"],
                             'mus': None,
                             'holding_potentials': [-70],
                             'distributions': None,
                             'conductance_types': ['passive', 'Ih_plateau', 'Ih_plateau2'],
                             'correlations': [0.0, 1.0]
                             }

hbp_population_params = {'input_type': 'distributed_delta',
                             'save_folder': 'hbp_population',
                             'name': 'hbp_population',
                         # 'cell_name': 'L4_BP_bAC217_1',
                             'cell_name': 'L5_TTPC2_cADpyr232_2',
                             'model_folder': join('..', '..', 'hbp_cell_models'),
                             'dt': dt,

                         'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                         'population_radius': population_radius,
                         'population_radii': population_radii,
                         'layer_5_thickness': layer_5_thickness,
                         'cut_off': cut_off,
                         'end_t': end_T,
                         'syn_tau': 2.,
                         'syntype': 'ExpSyn',
                         'syn_weight': 1e-3 / 100.,
                         'max_freq': 500,
                         'holding_potential': None,
                         'conductance_type': 'classic',
                         # 'lateral_electrode_parameters': lateral_electrode_parameters,
                         'center_electrode_parameters': center_elec_params,
                         'root_folder': root_folder,
                         'num_synapses': 1000,
                         'input_firing_rate': 5,
                         'input_regions': ['homogeneous', 'distal_tuft'],
                         'mus': None,
                         'holding_potentials': [None],
                         'distributions': None,
                         'conductance_types': ["Ih", "passive"],
                         'correlations': [0.0, 1.0]
                         }


stick_population_params = {'input_type': 'distributed_delta',
                             'name': 'stick_population',
                             'cell_name': 'stick_cell',
                             'dt': dt,
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': 0,
                             'cut_off': cut_off,
                             'end_t': end_T * 1,
                             # 'split_sim': 'top',
                             'syn_tau': dt * 3,
                             'syn_weight': 1e-3,
                             'max_freq': 500,
                             'holding_potential': -80,
                             'g_w_bar_scaling': 5.,
                             'conductance_type': 'generic',
                             'save_folder': 'stick_population_fork_test',
                             # 'lateral_electrode_parameters': stick_lateral_electrode_parameters,
                             'center_electrode_parameters': stick_center_electrode_parameters,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': ['top', 'homogeneous'],
                             'mus': [None, 2.0],
                             'distributions': ['increase'],
                             'correlations': [0.0, 1.0],
                             'correlation': 0.0,
                             }
multimorph_population_params = {'input_type': 'distributed_delta',
                             'name': 'multimorph_population',
                             'dt': dt,
                             'cell_name': 'multimorph',
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': layer_5_thickness,
                             'cut_off': cut_off,
                             'end_t': end_T,
                             'syn_tau': dt * 3,
                             'syn_weight': 0.03,
                             'max_freq': 500,
                             'holding_potential': -80,
                             'conductance_type': 'generic',
                             'save_folder': 'multimorph_population',
                             # 'lateral_electrode_parameters': lateral_electrode_parameters,
                             'center_electrode_parameters': center_elec_params,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': [
                                               'homogeneous',
                                               "distal_tuft",
                                               #"basal"
                                               ],
                             'mus': [2.0, None],
                             'distributions': [#'uniform',
                                               'linear_increase',
                                               #'linear_decrease'
                                              ],
                             'correlations': [0.0, 1.0]
                             }
asymmetry_params = {'input_type': 'distributed_asymmetry',
                            'dt': dt,
                            'cut_off': cut_off,
                            'end_t': end_T,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-3,
                            'max_freq': 500,
                            'holding_potential': -80,
                            'conductance_type': 'generic',
                            'asymmetry_fractions': [0.80],
                            'save_folder': 'asymmetry',
                            }

if __name__ == '__main__':
    from aPop.NeuralSimulation import NeuralSimulation
    generic_population_params.update({'input_region': 'distal_tuft',
                                      'mu': 0.0, 'correlation': 0.0,
                                      'cell_number': 0})
    ns = NeuralSimulation(**generic_population_params)
