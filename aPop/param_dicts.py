

# TODO: Is time resolution too low? Leski uses 2**-5. What happens if I increase it?

distributed_delta_params = {'input_type': 'distributed_delta',
                            'timeres_NEURON': 2**-4,
                            'timeres_python': 2**-4,
                            'cut_off': 200,
                            'end_t': 10000,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-4,
                            'repeats': None,
                            'max_freq': 500,
                            'holding_potential': -80,
                            'conductance_type': 'generic',
                            'save_folder': 'shape_function',
                            }


population_params = {'input_type': 'distributed_delta',
                            'timeres_NEURON': 2**-4,
                            'timeres_python': 2**-4,
                            'cut_off': 200,
                            'end_t': 1000,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-4,
                            'repeats': None,
                            'max_freq': 500,
                            'holding_potential': -80,
                            'conductance_type': 'generic',
                            'save_folder': 'population',
                            }


asymmetry_params = {'input_type': 'distributed_asymmetry',
                            'timeres_NEURON': 2**-4,
                            'timeres_python': 2**-4,
                            'cut_off': 200,
                            'end_t': 10000,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-4,
                            'repeats': None,
                            'max_freq': 500,
                            'holding_potential': -80,
                            'conductance_type': 'generic',
                            'asymmetry_fractions': [0.80],
                            'save_folder': 'asymmetry',
                            }