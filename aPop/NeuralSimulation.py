from __future__ import division
import os
import sys
from os.path import join
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False
from plotting_convention import mark_subplots, simplify_axes, LogNorm
import numpy as np
import pylab as plt
import neuron
import LFPy
import tools
h = neuron.h

class NeuralSimulation:
    def __init__(self, **kwargs):
        self.input_type = kwargs['input_type']
        self.input_region = kwargs['input_region']
        self.name = kwargs['name']
        self.cell_number = kwargs['cell_number'] if 'cell_number' in kwargs else None
        self.mu = kwargs['mu'] if 'mu' in kwargs else None
        self.distribution = kwargs['distribution'] if 'distribution' in kwargs else None
        self.correlation = kwargs['correlation'] if 'correlation' in kwargs else None

        self.param_dict = kwargs
        self.cell_name = kwargs['cell_name'] if 'cell_name' in kwargs else 'hay'
        self.max_freq = kwargs['max_freq']
        self.holding_potential = kwargs['holding_potential']
        self.conductance_type = kwargs['conductance_type']
        self.username = os.getenv('USER')

        self._set_input_params()
        self.sim_name = self.get_simulation_name()
        self.population_sim_name = self.sim_name[:-6]  # Just remove the cell number

        self.lateral_electrode_parameters = kwargs['lateral_electrode_parameters']
        self.elec_x_lateral = self.lateral_electrode_parameters['x']
        self.elec_y_lateral = self.lateral_electrode_parameters['y']
        self.elec_z_lateral = self.lateral_electrode_parameters['z']

        self.center_electrode_parameters = kwargs['center_electrode_parameters']
        self.elec_x_center = self.center_electrode_parameters['x']
        self.elec_y_center = self.center_electrode_parameters['y']
        self.elec_z_center = self.center_electrode_parameters['z']

        self.root_folder = kwargs['root_folder']
        self.neuron_models = join(self.root_folder, 'neuron_models')
        self.figure_folder = join(self.root_folder, self.param_dict['save_folder'])
        self.sim_folder = join(self.root_folder, self.param_dict['save_folder'], 'simulations')

        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
        if not os.path.isdir(self.sim_folder):
            os.mkdir(self.sim_folder)

        self.divide_into_welch = 8.
        self.welch_dict = {'Fs': 1000 / self.timeres_python,
                           'NFFT': int(self.num_tsteps/self.divide_into_welch),
                           'noverlap': int(self.num_tsteps/self.divide_into_welch/2.),
                           'window': plt.window_hanning,
                           'detrend': plt.detrend_mean,
                           'scale_by_freq': True,
                           }

    def get_simulation_name(self):
        if self.conductance_type == 'generic':
            conductance = '%s_%s_%1.1f' % (self.conductance_type, self.distribution, self.mu)
        else:
            if self.holding_potential is None:
                conductance = '%s_None' % (self.conductance_type)
            else:
                conductance = '%s_%+d' % (self.conductance_type, self.holding_potential)
        sim_name = '%s_%s_%s_%s_%1.2f_%05d' % (self.name, self.cell_name, self.input_region, conductance,
                                               self.correlation, self.cell_number)
        return sim_name

    def _set_input_params(self):
        self.timeres_NEURON = self.param_dict['timeres_NEURON']
        self.timeres_python = self.param_dict['timeres_python']
        self.cut_off = self.param_dict['cut_off']
        self.end_t = self.param_dict['end_t']
        self.max_freq = self.param_dict['max_freqs'] if 'max_freqs' in self.param_dict else 500
        self.num_tsteps = round(self.end_t/self.timeres_python + 1)

    def remove_active_mechanisms(self, remove_list):
        mt = h.MechanismType(0)
        for sec in h.allsec():
            for seg in sec:
                for mech in remove_list:
                    mt.select(mech)
                    mt.remove()

    def _return_cell(self, cell_x_y_z_rotation=None):

        if 'i_QA' not in neuron.h.__dict__.keys():
            print "",
            from suppress_print import suppress_stdout_stderr
            with suppress_stdout_stderr():
                neuron.load_mechanisms(join(self.neuron_models))
        cell = None
        if self.cell_name == 'hay':
            if self.conductance_type == 'generic':
                # print "Loading Hay"
                sys.path.append(join(self.neuron_models, 'hay'))
                from hay_active_declarations import active_declarations
                cell_params = {
                    'morphology': join(self.neuron_models, 'hay', 'cell1.hoc'),
                    'v_init': self.holding_potential,
                    'passive': False,           # switch on passive mechs
                    'nsegs_method': 'lambda_f',  # method for setting number of segments,
                    'lambda_f': 100,           # segments are isopotential at this frequency
                    'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
                    'timeres_python': self.timeres_python,
                    'tstartms': -self.cut_off,          # start time, recorders start at t=0
                    'tstopms': self.end_t,
                    'custom_code': [join(self.neuron_models, 'hay', 'custom_codes.hoc')],
                    'custom_fun': [active_declarations],  # will execute this function
                    'custom_fun_args': [{'conductance_type': self.conductance_type,
                                         'mu_factor': self.mu,
                                         'g_pas': 0.00005,#0.0002, # / 5,
                                         'distribution': self.distribution,
                                         'tau_w': 'auto',
                                         #'total_w_conductance': 6.23843378791,# / 5,
                                         'avrg_w_bar': 0.00005, # Half of "original"!!!
                                         'hold_potential': self.holding_potential}]
                }
                cell = LFPy.Cell(**cell_params)
            else:
                sys.path.append(join(self.neuron_models, 'hay'))
                from suppress_print import suppress_stdout_stderr
                with suppress_stdout_stderr():
                    neuron.load_mechanisms(join(self.neuron_models, 'hay', 'mod'))

                from hay_active_declarations import active_declarations
                cell_params = {
                    'morphology': join(self.neuron_models, 'hay', 'cell1.hoc'),
                    'v_init': self.holding_potential if self.holding_potential is not None else -70,
                    'passive': False,           # switch on passive mechs
                    'nsegs_method': 'lambda_f',  # method for setting number of segments,
                    'lambda_f': 100,           # segments are isopotential at this frequency
                    'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
                    'timeres_python': self.timeres_python,
                    'tstartms': -self.cut_off,          # start time, recorders start at t=0
                    'tstopms': self.end_t,
                    'custom_code': [join(self.neuron_models, 'hay', 'custom_codes.hoc')],
                    'custom_fun': [active_declarations],  # will execute this function
                    'custom_fun_args': [{'conductance_type': self.conductance_type,
                                         'hold_potential': self.holding_potential}]
                }
                cell = LFPy.Cell(**cell_params)
        elif 'L5_' in self.cell_name or 'L4_' in self.cell_name:
            remove_lists = {'active': [],
                            'passive': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                                        "SK_E2", "K_Tst", "K_Pst", "Im", "Ih",
                                        "CaDynamics_E2", "Ca_LVAst", "Ca"],
                            'Ih': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                                   "SK_E2", "K_Tst", "K_Pst", "Im",
                                   "CaDynamics_E2", "Ca_LVAst", "Ca"]}

            sys.path.append(self.param_dict['model_folder'])
            from suppress_print import suppress_stdout_stderr
            with suppress_stdout_stderr():
                # neuron.load_mechanisms(join(self.param_dict['model_folder'], 'mod'))
                from hbp_cells import return_cell
            cell_folder = join(self.param_dict['model_folder'], 'models', self.cell_name)
            cell = return_cell(cell_folder, self.end_t, self.timeres_NEURON, -self.cut_off)
            self.remove_active_mechanisms(remove_lists[self.conductance_type])

        elif self.cell_name == 'infinite_neurite':
            sys.path.append(join(self.neuron_models, 'infinite_neurite'))
            from infinite_neurite_active_declarations import active_declarations

            args = [{'mu_factor': self.param_dict['mu'],
                     'g_w_bar_scaling': self.param_dict['g_w_bar_scaling'],
                     'distribution': self.distribution,
                     }]
            cell_params = {
                'morphology': join(self.neuron_models, self.cell_name, 'infinite_neurite.hoc'),
                'v_init': self.holding_potential,
                'passive': False,
                'nsegs_method': None,
                'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
                'timeres_python': self.timeres_python,
                'tstartms': -self.cut_off,          # start time, recorders start at t=0
                'tstopms': self.end_t,
                'custom_fun': [active_declarations],
                'custom_fun_args': args,
            }
            cell = LFPy.Cell(**cell_params)
        else:
            raise NotImplementedError("Cell name is not recognized")
        if cell_x_y_z_rotation is not None:
            cell.set_rotation(z=cell_x_y_z_rotation[3])
            cell.set_pos(xpos=cell_x_y_z_rotation[0], ypos=cell_x_y_z_rotation[1], zpos=cell_x_y_z_rotation[2])
        return cell

    def save_neural_sim_single_input_data(self, cell):

        if self.name == 'vmem_3D_population':
            np.save(join(self.sim_folder, 'vmem_%s.npy' % self.sim_name), cell.vmem)
            np.save(join(self.sim_folder, 'xstart_%s.npy' % (self.sim_name)), cell.xstart)
            np.save(join(self.sim_folder, 'ystart_%s.npy' % (self.sim_name)), cell.ystart)
            np.save(join(self.sim_folder, 'zstart_%s.npy' % (self.sim_name)), cell.zstart)
            np.save(join(self.sim_folder, 'xend_%s.npy' % (self.sim_name)), cell.xend)
            np.save(join(self.sim_folder, 'yend_%s.npy' % (self.sim_name)), cell.yend)
            np.save(join(self.sim_folder, 'zend_%s.npy' % (self.sim_name)), cell.zend)
            np.save(join(self.sim_folder, 'xmid_%s.npy' % (self.sim_name)), cell.xmid)
            np.save(join(self.sim_folder, 'ymid_%s.npy' % (self.sim_name)), cell.ymid)
            np.save(join(self.sim_folder, 'zmid_%s.npy' % (self.sim_name)), cell.zmid)
            np.save(join(self.sim_folder, 'diam_%s.npy' % (self.sim_name)), cell.diam)
        else:
            name = self.sim_name
            if 'split_sim' in self.param_dict:
                name += '_%s' % self.param_dict['split_sim']
            lateral_electrode = LFPy.RecExtElectrode(cell, **self.lateral_electrode_parameters)
            lateral_electrode.calc_lfp()
            np.save(join(self.sim_folder, 'lateral_sig_%s.npy' % name), lateral_electrode.LFP)
            del lateral_electrode.LFP

            center_electrode = LFPy.RecExtElectrode(cell, **self.center_electrode_parameters)
            center_electrode.calc_lfp()
            np.save(join(self.sim_folder, 'center_sig_%s.npy' % name), center_electrode.LFP)
            del center_electrode.LFP
            
            if self.cell_number == 0:
                np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
                if hasattr(cell, 'vmem'):
                    np.save(join(self.sim_folder, 'vmem_%s.npy' % name), cell.vmem)
                np.save(join(self.sim_folder, 'imem_%s.npy' % name), cell.imem)
                np.save(join(self.sim_folder, 'synidx_%s.npy' % name), cell.synidx)

                np.save(join(self.sim_folder, 'lateral_elec_x_%s.npy' % self.cell_name), lateral_electrode.x)
                np.save(join(self.sim_folder, 'lateral_elec_y_%s.npy' % self.cell_name), lateral_electrode.y)
                np.save(join(self.sim_folder, 'lateral_elec_z_%s.npy' % self.cell_name), lateral_electrode.z)

                np.save(join(self.sim_folder, 'center_elec_x_%s.npy' % self.cell_name), center_electrode.x)
                np.save(join(self.sim_folder, 'center_elec_y_%s.npy' % self.cell_name), center_electrode.y)
                np.save(join(self.sim_folder, 'center_elec_z_%s.npy' % self.cell_name), center_electrode.z)

                np.save(join(self.sim_folder, 'xstart_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.xstart)
                np.save(join(self.sim_folder, 'ystart_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.ystart)
                np.save(join(self.sim_folder, 'zstart_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.zstart)
                np.save(join(self.sim_folder, 'xend_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.xend)
                np.save(join(self.sim_folder, 'yend_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.yend)
                np.save(join(self.sim_folder, 'zend_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.zend)
                np.save(join(self.sim_folder, 'xmid_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.xmid)
                np.save(join(self.sim_folder, 'ymid_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.ymid)
                np.save(join(self.sim_folder, 'zmid_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.zmid)
                np.save(join(self.sim_folder, 'diam_%s_%s.npy' %
                             (self.cell_name, self.param_dict['conductance_type'])), cell.diam)

    def run_distributed_synaptic_simulation(self):

        # if os.path.isfile(join(self.sim_folder, 'center_sig_%s.npy' % self.sim_name)) or \
        #    os.path.isfile(join(self.sim_folder, 'vmem_%s.npy' % self.sim_name)):
        #     print "Skipping ", self.sim_name
        #     return

        plt.seed(123 * self.cell_number)
        if 'shape_function' in self.name:
            x_y_z_rot = np.array([0, 0, 0, 2*np.pi*np.random.random()])
        elif 'population' in self.name:
            x_y_z_rot = np.load(os.path.join(self.param_dict['root_folder'],
                                             self.param_dict['save_folder'],
                             'x_y_z_rot_%s.npy' % self.param_dict['name']))[self.cell_number]
        else:
            raise RuntimeError("Something wrong with simulation set-up!")

        cell = self._return_cell(x_y_z_rot)
        cell, syn = self._make_distributed_synaptic_stimuli(cell)

        rec_vmem = True if self.name is 'vmem_3D_population' else False
        cell.simulate(rec_imem=True, rec_vmem=rec_vmem)

        # [plt.plot(cell.tvec, cell.vmem[idx, :]) for idx in range(cell.totnsegs)]
        # plt.plot(cell.tvec, cell.vmem[1, :])
        # plt.show()
        self.save_neural_sim_single_input_data(cell)
        if ('shape_function' in self.name) or (self.cell_number < 5):
            self._draw_all_elecs_with_distance(cell)


    def run_asymmetry_simulation(self, mu, fraction, distribution, cell_number):
        plt.seed(123 * cell_number)
        # if os.path.isfile(join(self.sim_folder, 'sig_%s.npy' % sim_name)):
        #     print "Skipping ", mu, input_sec, distribution, tau_w, weight, 'sig_%s.npy' % sim_name
        #     return

        electrode = LFPy.RecExtElectrode(**self.lateral_electrode_parameters)
        cell = self._return_cell(mu, distribution)
        cell, syn_apic, syn_basal = self._make_asymmetry_distributed_synaptic_stimuli(cell, fraction)
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        self.save_neural_sim_single_input_data(cell, electrode, 'asymmetry_%1.2f' % fraction, mu, distribution, cell_number)
        self._draw_all_elecs_with_distance(cell, electrode, 'asymmetry_%1.2f' % fraction, mu, distribution, cell_number)

    def _make_distributed_synaptic_stimuli(self, cell):

        syntype = 'ExpSynI' if not 'syntype' in self.param_dict else self.param_dict['syntype']

        # Define synapse parameters
        synapse_params = {
            'e': 0.,                   # reversal potential (not used for ExpSynI)
            'syntype': syntype,       # synapse type
            'tau': self.param_dict['syn_tau'],                # syn. time constant
            'weight': self.param_dict['syn_weight'],            # syn. weight
            'record_current': False,
            'color': 'r',
        }
        inhibitory_synapse_params = {
            'e': 0.,                   # reversal potential (not used for ExpSynI)
            'syntype': syntype,       # synapse type
            'tau': self.param_dict['syn_tau'],                # syn. time constant
            'weight': self.param_dict['inhibitory_syn_weight'],            # syn. weight
            'record_current': False,
            'color': 'b',
        }

        if self.input_region == 'distal_tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 900
        elif self.input_region == 'tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 600
        elif self.input_region == 'homogeneous':
            input_pos = ['apic', 'dend']
            maxpos = 10000
            minpos = -10000
        elif self.input_region == 'basal':
            input_pos = ['dend']
            maxpos = 10000
            minpos = -10000
        elif self.input_region == 'perisomatic_inhibition':
            inhibitory_input_pos = ['dend']
            inhibitory_maxpos = 10000
            inhibitory_minpos = -10000

        elif self.input_region == 'balanced':
            inhibitory_input_pos = ['dend']
            inhibitory_maxpos = 10000
            inhibitory_minpos = -10000

            excitatory_input_pos = ['apic', 'dend']
            excitatory_maxpos = 10000
            excitatory_minpos = -10000

        elif self.input_region == 'top':
            input_pos = ['dend']
            maxpos = 10000
            minpos = 500
        elif self.input_region == 'bottom':
            input_pos = ['dend']
            maxpos = 500
            minpos = -1000
        elif 'dual' in self.input_region:
            input_pos_tuft = ['apic']
            maxpos_tuft = 10000
            minpos_tuft = 900
            input_pos_homo = ['apic', 'dend']
            maxpos_homo = 10000
            minpos_homo = -10000
            dual_fraction = float(self.input_region[4:])
            print "Dual fraction: ", dual_fraction
        else:
            raise RuntimeError("Use other input section")

        num_synapses = self.param_dict['num_synapses']
        firing_rate = self.param_dict['input_firing_rate']
        if 'dual' in self.input_region:
            num_synapses_tuft = int(num_synapses * dual_fraction)
            num_synapses_homo = num_synapses - num_synapses_tuft
            cell_input_idxs_tuft = cell.get_rand_idx_area_norm(section=input_pos_tuft, nidx=num_synapses_tuft,
                                                               z_min=minpos_tuft, z_max=maxpos_tuft)
            cell_input_idxs_homo = cell.get_rand_idx_area_norm(section=input_pos_homo, nidx=num_synapses_homo,
                                                               z_min=minpos_homo, z_max=maxpos_homo)
            cell_input_idxs = np.r_[cell_input_idxs_homo, cell_input_idxs_tuft]

        elif self.input_region == 'balanced':

            cell_input_idxs_in = cell.get_rand_idx_area_norm(section=inhibitory_input_pos,
                                                             nidx=self.param_dict['num_inhibitory_synapses'],
                                                             z_min=inhibitory_minpos,
                                                             z_max=inhibitory_maxpos)

            cell_input_idxs_ex = cell.get_rand_idx_area_norm(section=excitatory_input_pos,
                                                             nidx=self.param_dict['num_synapses'],
                                                             z_min=excitatory_minpos,
                                                             z_max=excitatory_maxpos)
        elif self.input_region == 'perisomatic_inhibition':

            cell_input_idxs_in = cell.get_rand_idx_area_norm(section=inhibitory_input_pos,
                                                             nidx=self.param_dict['num_inhibitory_synapses'],
                                                             z_min=inhibitory_minpos,
                                                             z_max=inhibitory_maxpos)

        else:
            cell_input_idxs = cell.get_rand_idx_area_norm(section=input_pos,
                                                          nidx=num_synapses, z_min=minpos, z_max=maxpos)

        if self.correlation < 1e-6:
            if self.input_region == 'balanced':
                spike_trains_in = LFPy.inputgenerators.stationary_poisson(
                    self.param_dict['num_inhibitory_synapses'], self.param_dict['inhibitory_input_firing_rate'],
                    cell.tstartms, cell.tstopms)
                spike_trains_ex = LFPy.inputgenerators.stationary_poisson(
                    self.param_dict['num_synapses'], self.param_dict['input_firing_rate'],
                    cell.tstartms, cell.tstopms)

                synapses_ex = self.set_input_spiketrain(cell, cell_input_idxs_ex,
                                                        spike_trains_ex, synapse_params)
                synapses_in = self.set_input_spiketrain(cell, cell_input_idxs_in,
                                                        spike_trains_in, inhibitory_synapse_params)

                synapses = synapses_ex + synapses_in
            elif self.input_region == 'perisomatic_inhibition':
                spike_trains_in = LFPy.inputgenerators.stationary_poisson(
                    self.param_dict['num_inhibitory_synapses'], self.param_dict['inhibitory_input_firing_rate'],
                    cell.tstartms, cell.tstopms)


                synapses_in = self.set_input_spiketrain(cell, cell_input_idxs_in,
                                                        spike_trains_in, inhibitory_synapse_params)

                synapses = synapses_in
            else:
                spike_trains = LFPy.inputgenerators.stationary_poisson(num_synapses,
                                                                   firing_rate, cell.tstartms, cell.tstopms)
                synapses = self.set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)

        else:
            if self.input_region == 'balanced':
                all_spiketimes_ex = np.load(join(self.param_dict['root_folder'], self.param_dict['save_folder'],
                                 'all_spike_trains_%s.npy' % self.param_dict['name'])).item()
                all_spiketimes_in = np.load(join(self.param_dict['root_folder'], self.param_dict['save_folder'],
                                 'all_inhibitory_spike_trains_%s.npy' % self.param_dict['name'])).item()
                spike_train_idxs_ex = np.random.choice(np.arange(int(num_synapses/self.correlation)),
                                                    num_synapses, replace=False)

                spike_train_idxs_in = np.random.choice(
                    np.arange(int(self.param_dict['num_inhibitory_synapses']/self.correlation)),
                    self.param_dict['num_inhibitory_synapses'], replace=False)

                spike_trains_ex = [all_spiketimes_ex[idx] for idx in spike_train_idxs_ex]
                spike_trains_in = [all_spiketimes_in[idx] for idx in spike_train_idxs_in]

                synapses_ex = self.set_input_spiketrain(cell, cell_input_idxs_ex,
                                                        spike_trains_ex, synapse_params)
                synapses_in = self.set_input_spiketrain(cell, cell_input_idxs_in,
                                                        spike_trains_in, inhibitory_synapse_params)
                synapses = synapses_ex + synapses_in
            elif self.input_region == 'perisomatic_inhibition':

                all_spiketimes_in = np.load(join(self.param_dict['root_folder'], self.param_dict['save_folder'],
                                 'all_inhibitory_spike_trains_%s.npy' % self.param_dict['name'])).item()

                spike_train_idxs_in = np.random.choice(
                    np.arange(int(self.param_dict['num_inhibitory_synapses']/self.correlation)),
                    self.param_dict['num_inhibitory_synapses'], replace=False)

                spike_trains_in = [all_spiketimes_in[idx] for idx in spike_train_idxs_in]

                synapses_in = self.set_input_spiketrain(cell, cell_input_idxs_in,
                                                        spike_trains_in, inhibitory_synapse_params)
                synapses = synapses_in
            else:
                all_spiketimes = np.load(join(self.param_dict['root_folder'], self.param_dict['save_folder'],
                                 'all_spike_trains_%s.npy' % self.param_dict['name'])).item()

                spike_train_idxs = np.random.choice(np.arange(int(num_synapses/self.correlation)),
                                                    num_synapses, replace=False)
                spike_trains = [all_spiketimes[idx] for idx in spike_train_idxs]
                synapses = self.set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)

        return cell, synapses

    def _make_asymmetry_distributed_synaptic_stimuli(self, cell, fraction):

        # Define synapse parameters
        synapse_params = {
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': self.param_dict['syn_tau'],                # syn. time constant
            'weight': self.param_dict['syn_weight'],            # syn. weight
            'record_current': False,
        }

        num_synapses_apic = int(1000 * fraction)
        input_pos = ['apic']
        maxpos = 10000
        minpos = 600
        cell_input_idxs_apic = cell.get_rand_idx_area_norm(section=input_pos, nidx=num_synapses_apic,
                                                      z_min=minpos, z_max=maxpos)
        spike_trains_apic = LFPy.inputgenerators.stationary_poisson(num_synapses_apic, 5, cell.tstartms, cell.tstopms)
        synapses_apic = self.set_input_spiketrain(cell, cell_input_idxs_apic, spike_trains_apic, synapse_params)

        num_synapses_basal = 1000 - num_synapses_apic
        input_pos = ['apic', 'dend']
        maxpos = 600
        minpos = -10000

        if num_synapses_apic + num_synapses_basal != 1000:
            print num_synapses_basal, num_synapses_apic
            raise RuntimeError("Does not sum to 1000")

        cell_input_idxs_basal = cell.get_rand_idx_area_norm(section=input_pos, nidx=num_synapses_basal,
                                                      z_min=minpos, z_max=maxpos)
        spike_trains_basal = LFPy.inputgenerators.stationary_poisson(num_synapses_basal, 5, cell.tstartms, cell.tstopms)
        synapses_basal = self.set_input_spiketrain(cell, cell_input_idxs_basal, spike_trains_basal, synapse_params)

        return cell, synapses_apic, synapses_basal

    def set_input_spiketrain(self, cell, cell_input_idxs, spike_trains, synapse_params):
        synapse_list = []
        for number, comp_idx in enumerate(cell_input_idxs):
            synapse_params.update({'idx': int(comp_idx)})
            if 'split_sim' in self.param_dict:
                if self.param_dict['split_sim'] == 'top' and cell.zmid[comp_idx] < 500:
                    continue
                if self.param_dict['split_sim'] == 'bottom' and cell.zmid[comp_idx] > 500:
                    continue

            s = LFPy.Synapse(cell, **synapse_params)
            s.set_spike_times(spike_trains[number])
            synapse_list.append(s)
        return synapse_list

    def _return_elec_subplot_number_with_distance(self, elec_number):
        """ Return the subplot number for the distance study
        """
        num_elec_cols = len(set(self.elec_x_lateral))
        num_elec_rows = len(set(self.elec_z_lateral))
        elec_idxs = np.arange(len(self.elec_x_lateral)).reshape(num_elec_rows, num_elec_cols)
        row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
        num_plot_cols = num_elec_cols + 3
        plot_number = row * num_plot_cols + col + 4
        return plot_number

    def _return_elec_row_col(self, elec_number):
        """ Return the subplot number for the distance study
        """
        num_elec_cols = len(set(self.elec_x_lateral))
        num_elec_rows = len(set(self.elec_z_lateral))
        elec_idxs = np.arange(len(self.elec_x_lateral)).reshape(num_elec_rows, num_elec_cols)
        row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
        return row, col

    def plot_cell_to_ax(self, cell, ax):
        [ax.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
                 lw=1.5, color='0.5', zorder=0, alpha=1)
            for idx in xrange(len(cell.xmid))]

        # [ax.plot(cell.xmid[idx], cell.zmid[idx], 'g.', ms=4) for idx in cell.synidx]
        for syn in cell.synapses:
            ax.plot(cell.xmid[syn.idx], cell.zmid[syn.idx], syn.color, marker='.', ms=4)

    def _draw_all_elecs_with_distance(self, cell):

        plt.close('all')
        fig = plt.figure(figsize=[18, 9])
        fig.subplots_adjust(right=0.97, left=0.05)
        apic_clr = 'orange'
        soma_clr = 'olive'
        middle_clr = 'm'
        soma_idx = 0
        apic_idx = np.argmin(np.abs(cell.zmid - 1000))
        middle_idx = np.argmin(np.abs(cell.zmid - 500))

        plot_psd = True

        if plot_psd:
            scale = 'log'
            # v_x_apic, [v_y_apic] = tools.return_freq_and_psd_welch(cell.vmem[apic_idx], self.welch_dict)
            # v_x_middle, [v_y_middle] = tools.return_freq_and_psd_welch(cell.vmem[middle_idx], self.welch_dict)
            # v_x_soma, [v_y_soma] = tools.return_freq_and_psd_welch(cell.vmem[soma_idx], self.welch_dict)
            # i_x_apic, [i_y_apic] = tools.return_freq_and_psd_welch(cell.imem[apic_idx], self.welch_dict)
            # i_x_middle, [i_y_middle] = tools.return_freq_and_psd_welch(cell.imem[middle_idx], self.welch_dict)
            # i_x_soma, [i_y_soma] = tools.return_freq_and_psd_welch(cell.imem[soma_idx], self.welch_dict)

            # v_x_apic, [v_y_apic] = tools.return_freq_and_psd(self.timeres_python/1000., cell.vmem[apic_idx])
            # v_x_middle, [v_y_middle] = tools.return_freq_and_psd(self.timeres_python/1000., cell.vmem[middle_idx])
            # v_x_soma, [v_y_soma] = tools.return_freq_and_psd(self.timeres_python/1000., cell.vmem[soma_idx])
            # i_x_apic, [i_y_apic] = tools.return_freq_and_psd(self.timeres_python/1000., cell.imem[apic_idx])
            # i_x_middle, [i_y_middle] = tools.return_freq_and_psd(self.timeres_python/1000., cell.imem[middle_idx])
            # i_x_soma, [i_y_soma] = tools.return_freq_and_psd(self.timeres_python/1000., cell.imem[soma_idx])
        else:
            scale = 'linear'
        #     v_x_apic, v_y_apic = cell.tvec, cell.vmem[apic_idx]
        #     v_x_middle, v_y_middle = cell.tvec, cell.vmem[middle_idx]
        #     v_x_soma, v_y_soma = cell.tvec, cell.vmem[soma_idx]
        #     i_x_apic, i_y_apic = cell.tvec, cell.imem[apic_idx]
        #     i_x_middle, i_y_middle = cell.tvec, cell.imem[middle_idx]
        #     i_x_soma, i_y_soma = cell.tvec, cell.imem[soma_idx]

        ax_top = fig.add_subplot(3, 6, 4, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_top_n = fig.add_subplot(3, 6, 5, xlim=[1, self.max_freq], xscale=scale, yscale=scale, ylim=[1e-3, 1e0])
        ax_mid = fig.add_subplot(3, 6, 10, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_mid_n = fig.add_subplot(3, 6, 11, xlim=[1, self.max_freq], xscale=scale, yscale=scale, ylim=[1e-3, 1e0])
        ax_bottom = fig.add_subplot(3, 6, 16, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_bottom_n = fig.add_subplot(3, 6, 17, xlim=[1, self.max_freq], xscale=scale, yscale=scale, ylim=[1e-3, 1e0])
        cell_ax = fig.add_subplot(1, 6, 3, aspect=1, frameon=False, xticks=[], yticks=[])
        ax_center = fig.add_subplot(2, 6, 6, xlim=[1, self.max_freq], xscale=scale, xlabel='Hz', ylabel='z')
        ax_somav = fig.add_subplot(2, 6, 12, xlabel='Time', ylabel='V$_m$')
        ax_somav.plot(cell.tvec, cell.somav)

        self.plot_cell_to_ax(cell, cell_ax)
        center_electrode = LFPy.RecExtElectrode(cell, **self.center_electrode_parameters)
        center_electrode.calc_lfp()
        center_LFP = 1000 * center_electrode.LFP

        # freq, psd = tools.return_freq_and_psd(self.timeres_python/1000., center_LFP)
        freq, psd = tools.return_freq_and_psd_welch(center_LFP, self.welch_dict)
        vmax = 10**np.ceil(np.log10(np.max(psd)))
        im_dict = {'cmap': 'hot', 'norm': LogNorm(), 'vmin': vmax/1e4, 'vmax': vmax}
        img = ax_center.pcolormesh(freq, center_electrode.z, psd, **im_dict)
        cbar = plt.colorbar(img, ax=ax_center)
        del center_electrode
        del center_LFP
        all_elec_ax = [ax_top, ax_mid, ax_bottom]
        all_elec_n_ax = [ax_top_n, ax_mid_n, ax_bottom_n]

        # apic_title = '%1.2f (%1.2f) mV' % (np.mean(cell.vmem[apic_idx]), np.std(cell.vmem[apic_idx]))
        # middle_title = '%1.2f (%1.2f) mV' % (np.mean(cell.vmem[middle_idx]), np.std(cell.vmem[middle_idx]))
        # soma_title = '%1.2f (%1.2f) mV' % (np.mean(cell.vmem[soma_idx]), np.std(cell.vmem[soma_idx]))

        # ax_v_apic = fig.add_subplot(3, 6, 1, xlim=[1, self.max_freq], ylabel='mV$^2$/Hz',
        #                             title=apic_title, xscale=scale, yscale=scale)
        # ax_v_middle = fig.add_subplot(3, 6, 7, xlim=[1, self.max_freq], ylabel='mV$^2$/Hz',
        #                               title=middle_title, xscale=scale, yscale=scale)
        # ax_v_soma = fig.add_subplot(3, 6, 13, xlim=[1, self.max_freq], ylabel='mV$^2$/Hz',
        #                             title=soma_title, xscale=scale, yscale=scale)

        # ax_i_apic = fig.add_subplot(3, 6, 2, xlim=[1, self.max_freq], ylabel='nA$^2$/Hz', xscale=scale, yscale=scale)
        # ax_i_middle = fig.add_subplot(3, 6, 8, xlim=[1, self.max_freq], ylabel='nA$^2$/Hz', xscale=scale, yscale=scale)
        # ax_i_soma = fig.add_subplot(3, 6, 14, xlim=[1, self.max_freq], ylabel='nA$^2$/Hz', xscale=scale, yscale=scale)

        # average_over = 4
        # smooth_v_x_apic = v_x_apic[::average_over]
        # max_freq_idx = np.argmin(np.abs(freq - self.max_freq))

        # smooth_v_y_apic = tools.smooth_signal(smooth_v_x_apic, v_x_apic, v_y_apic)
        # smooth_v_x_middle = v_x_middle[::average_over]
        # smooth_v_y_middle = tools.smooth_signal(smooth_v_x_middle, v_x_middle, v_y_middle)

        # smooth_v_x_soma = v_x_soma[::average_over]
        # smooth_v_y_soma = tools.smooth_signal(smooth_v_x_soma, v_x_soma, v_y_soma)

        # smooth_i_x_apic = v_x_apic[::average_over]
        # smooth_i_y_apic = tools.smooth_signal(smooth_i_x_apic, i_x_apic, i_y_apic)

        # smooth_i_x_middle = v_x_middle[::average_over]
        # smooth_i_y_middle = tools.smooth_signal(smooth_i_x_middle, i_x_middle, i_y_middle)

        # smooth_i_x_soma = v_x_soma[::average_over]
        # smooth_i_y_soma = tools.smooth_signal(smooth_i_x_soma, i_x_soma, i_y_soma)

        # ax_v_apic.loglog(v_x_apic, v_y_apic, c=apic_clr, lw=0.2, alpha=0.5)
        # ax_v_middle.loglog(v_x_middle, v_y_middle, c=middle_clr, lw=0.2, alpha=0.5)
        # ax_v_soma.loglog(v_x_soma, v_y_soma, c=soma_clr, lw=0.2, alpha=0.5)

        # ax_i_apic.loglog(i_x_apic, i_y_apic, c=apic_clr, lw=0.2, alpha=0.5)
        # ax_i_middle.loglog(i_x_middle, i_y_middle, c=middle_clr, lw=0.2, alpha=0.5)
        # ax_i_soma.loglog(i_x_soma, i_y_soma, c=soma_clr, lw=0.2, alpha=0.5)

        # ax_v_apic.loglog(smooth_v_x_apic[1:max_freq_idx], smooth_v_y_apic[1:max_freq_idx], c=apic_clr)
        # ax_v_middle.loglog(smooth_v_x_middle[1:max_freq_idx], smooth_v_y_middle[1:max_freq_idx], c=middle_clr)
        # ax_v_soma.loglog(smooth_v_x_soma[1:max_freq_idx], smooth_v_y_soma[1:max_freq_idx], c=soma_clr)
        #
        # ax_i_apic.loglog(smooth_i_x_apic[1:max_freq_idx], smooth_i_y_apic[1:max_freq_idx], c=apic_clr)
        # ax_i_middle.loglog(smooth_i_x_middle[1:max_freq_idx], smooth_i_y_middle[1:max_freq_idx], c=middle_clr)
        # ax_i_soma.loglog(smooth_i_x_soma[1:max_freq_idx], smooth_i_y_soma[1:max_freq_idx], c=soma_clr)

        # cell_ax.plot(cell.xmid[apic_idx], cell.zmid[apic_idx], 'D', ms=10, c=apic_clr)
        # cell_ax.plot(cell.xmid[middle_idx], cell.zmid[middle_idx], 'D', ms=10, c=middle_clr)
        # cell_ax.plot(cell.xmid[soma_idx], cell.zmid[soma_idx], 'D', ms=10, c=soma_clr)
        # if self.conductance_type == 'generic':
        #     sim_name = '%s_%s_%s_%s_%+1.1f_%1.5fuS_%05d' % (self.cell_name, self.input_type,
        #                                                 input_region, distribution, mu,
        #                                                 self.param_dict['syn_weight'], cell_number)
        # else:
        #      sim_name = '%s_%s_%s_%s_%+d_%1.5fuS_%05d' % (self.cell_name, self.input_type, input_region,
        #                                               self.conductance_type, self.holding_potential, self.param_dict['syn_weight'], cell_number)
        electrode = LFPy.RecExtElectrode(cell, **self.lateral_electrode_parameters)
        electrode.calc_lfp()
        fig.suptitle(self.sim_name)
        LFP = 1000 * electrode.LFP#np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))[:, :]

        if self.input_type == 'distributed_delta':
            freqs, sig_psd = tools.return_freq_and_psd_welch(LFP, self.welch_dict)
            # freqs, sig_psd = tools.return_freq_and_psd(self.timeres_python/1000., LFP)
        else:
            # freqs, sig_psd = tools.return_freq_and_psd(self.timeres_python/1000., LFP)
            freqs, sig_psd = tools.return_freq_and_psd_welch(LFP, self.welch_dict)

        cutoff_dist = 1000
        dist_clr = lambda dist: plt.cm.Greys(np.log(dist) / np.log(cutoff_dist))

        for elec in xrange(len(self.elec_z_lateral)):
            if self.elec_x_lateral[elec] > cutoff_dist:
                continue
            clr = dist_clr(self.elec_x_lateral[elec])
            cell_ax.plot(self.elec_x_lateral[elec], self.elec_z_lateral[elec], 'o', c=clr)

            row, col = self._return_elec_row_col(elec)
            # smooth_freqs = freqs[::average_over]
            # smooth_sig = tools.smooth_signal(smooth_freqs, freqs, sig_psd[elec, :])
            # smooth_sig = tools.smooth_signal(smooth_freqs, freqs, sig_psd[elec, :])

            all_elec_ax[row].loglog(freqs, sig_psd[elec, :], color=clr)
            all_elec_n_ax[row].loglog(freqs, sig_psd[elec, :] / np.max(sig_psd[elec, 1:]), color=clr)

            # all_elec_ax[row].loglog(smooth_freqs[1:max_freq_idx], smooth_sig[1:max_freq_idx], color=clr, lw=1)
            # all_elec_n_ax[row].loglog(smooth_freqs[1:max_freq_idx], smooth_sig[1:max_freq_idx] / np.max(smooth_sig[1:max_freq_idx]), color=clr, lw=1)

        [ax.grid(True) for ax in all_elec_ax + all_elec_n_ax ]
         #[ax_i_apic, ax_i_middle, ax_i_soma, ax_v_apic, ax_v_middle, ax_v_soma]]

        fig.savefig(join(self.figure_folder, '%s.png' % self.sim_name))

    def plot_LFP_with_distance(self, fraction):
        print "plotting "
        plt.seed(123*self.cell_number)

        cell = self._return_cell()
        cell, syn_apic, syn_basal = self._make_asymmetry_distributed_synaptic_stimuli(cell, fraction)
        cell.imem = np.load(join(self.sim_folder, 'imem_%s.npy' % self.sim_name))
        cell.vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % self.sim_name))
        plt.close('all')
        fig = plt.figure(figsize=[18, 9])
        fig.subplots_adjust(right=0.97, left=0.05)
        apic_clr = 'orange'
        soma_clr = 'olive'
        middle_clr = 'm'
        soma_idx = 0
        apic_idx = np.argmin(np.abs(cell.zmid - 1000))
        middle_idx = np.argmin(np.abs(cell.zmid - 500))

        plot_psd = True
        if plot_psd:
            scale = 'log'
            v_x_apic, [v_y_apic] = tools.return_freq_and_psd_welch(cell.vmem[apic_idx], self.welch_dict)
            v_x_middle, [v_y_middle] = tools.return_freq_and_psd_welch(cell.vmem[middle_idx], self.welch_dict)
            v_x_soma, [v_y_soma] = tools.return_freq_and_psd_welch(cell.vmem[soma_idx], self.welch_dict)
            i_x_apic, [i_y_apic] = tools.return_freq_and_psd_welch(cell.imem[apic_idx], self.welch_dict)
            i_x_middle, [i_y_middle] = tools.return_freq_and_psd_welch(cell.imem[middle_idx], self.welch_dict)
            i_x_soma, [i_y_soma] = tools.return_freq_and_psd_welch(cell.imem[soma_idx], self.welch_dict)
        else:
            scale ='linear'
            v_x_apic, v_y_apic = cell.tvec, cell.vmem[apic_idx]
            v_x_middle, v_y_middle = cell.tvec, cell.vmem[middle_idx]
            v_x_soma, v_y_soma = cell.tvec, cell.vmem[soma_idx]
            i_x_apic, i_y_apic = cell.tvec, cell.imem[apic_idx]
            i_x_middle, i_y_middle = cell.tvec, cell.imem[middle_idx]
            i_x_soma, i_y_soma = cell.tvec, cell.imem[soma_idx]
        ylim= [1e-5, 1]
        ax_top = fig.add_subplot(3, 5, 4, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_top_n = fig.add_subplot(3, 5, 5, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_mid = fig.add_subplot(3, 5, 9, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_mid_n = fig.add_subplot(3, 5, 10, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_bottom = fig.add_subplot(3, 5, 14, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_bottom_n = fig.add_subplot(3, 5, 15, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        cell_ax = fig.add_subplot(1, 5, 3, aspect=1, frameon=False, xticks=[], yticks=[])
        self.plot_cell_to_ax(cell, cell_ax)

        all_elec_ax = [ax_top, ax_mid, ax_bottom]
        all_elec_n_ax = [ax_top_n, ax_mid_n, ax_bottom_n]

        apic_title = '%1.2f (%1.2f) mV' % (np.mean(cell.vmem[apic_idx]), np.std(cell.vmem[apic_idx]))
        middle_title = '%1.2f (%1.2f) mV' % (np.mean(cell.vmem[middle_idx]), np.std(cell.vmem[middle_idx]))
        soma_title = '%1.2f (%1.2f) mV' % (np.mean(cell.vmem[soma_idx]), np.std(cell.vmem[soma_idx]))

        ax_v_apic = fig.add_subplot(3, 5, 1, xlim=[1, self.max_freq], ylabel='mV$^2$/Hz',
                                    title=apic_title, xscale=scale, yscale=scale)
        ax_v_middle = fig.add_subplot(3, 5, 6, xlim=[1, self.max_freq], ylabel='mV$^2$/Hz',
                                      title=middle_title, xscale=scale, yscale=scale)
        ax_v_soma = fig.add_subplot(3, 5, 11, xlim=[1, self.max_freq], ylabel='mV$^2$/Hz',
                                    title=soma_title, xscale=scale, yscale=scale)

        ax_i_apic = fig.add_subplot(3, 5, 2, xlim=[1, self.max_freq], ylabel='nA$^2$/Hz', xscale=scale, yscale=scale)
        ax_i_middle = fig.add_subplot(3, 5, 7, xlim=[1, self.max_freq], ylabel='nA$^2$/Hz', xscale=scale, yscale=scale)
        ax_i_soma = fig.add_subplot(3, 5, 12, xlim=[1, self.max_freq], ylabel='nA$^2$/Hz', xscale=scale, yscale=scale)

        ax_v_apic.loglog(v_x_apic, v_y_apic, c=apic_clr)
        ax_v_middle.loglog(v_x_middle, v_y_middle, c=middle_clr)
        ax_v_soma.loglog(v_x_soma, v_y_soma, c=soma_clr)

        ax_i_apic.loglog(i_x_apic, i_y_apic, c=apic_clr)
        ax_i_middle.loglog(i_x_middle, i_y_middle, c=middle_clr)
        ax_i_soma.loglog(i_x_soma, i_y_soma, c=soma_clr)

        cell_ax.plot(cell.xmid[apic_idx], cell.zmid[apic_idx], 'D', ms=10, c=apic_clr)
        cell_ax.plot(cell.xmid[middle_idx], cell.zmid[middle_idx], 'D', ms=10, c=middle_clr)
        cell_ax.plot(cell.xmid[soma_idx], cell.zmid[soma_idx], 'D', ms=10, c=soma_clr)

        fig.suptitle(self.sim_name)
        LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s.npy' % self.sim_name))[:, :]

        freqs, sig_psd = tools.return_freq_and_psd_welch(LFP, self.welch_dict)

        cutoff_dist = 1000
        dist_clr = lambda dist: plt.cm.Greys(np.log(dist) / np.log(cutoff_dist))

        for elec in xrange(len(self.elec_z_lateral)):
            if self.elec_x_lateral[elec] > cutoff_dist:
                continue
            clr = dist_clr(self.elec_x_lateral[elec])
            cell_ax.plot(self.elec_x_lateral[elec], self.elec_z_lateral[elec], 'o', c=clr)

            row, col = self._return_elec_row_col(elec)
            all_elec_ax[row].loglog(freqs, sig_psd[elec, :], color=clr, lw=1)
            all_elec_n_ax[row].loglog(freqs, sig_psd[elec, :] / np.max(sig_psd[elec, :]), color=clr, lw=1)

        [ax.grid(True) for ax in all_elec_ax + all_elec_n_ax +
         [ax_i_apic, ax_i_middle, ax_i_soma, ax_v_apic, ax_v_middle, ax_v_soma]]

        for ax in all_elec_ax + all_elec_n_ax + [ax_i_apic, ax_i_middle, ax_i_soma, ax_v_apic, ax_v_middle, ax_v_soma]:
            max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()) for l in ax.get_lines()])))
            ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])

        fig.savefig(join(self.figure_folder, 'LFP_%s.png' % self.sim_name))

    def plot_F(self, freqs, sig_psd):
        print "Plotting shape function"
        plt.close('all')
        fig = plt.figure(figsize=[18, 9])
        fig.subplots_adjust(right=0.97, left=0.05)
        plt.seed(123*0)

        scale = 'log'
        cell_ax = fig.add_subplot(1, 3, 1, aspect=1, frameon=False, xticks=[], yticks=[])

        name = '%s_%s' % (self.cell_name, self.conductance_type)
        if self.conductance_type == 'generic':
            xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % name))
            zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % name))
            xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % name))
            zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % name))
            xend = np.load(join(self.sim_folder, 'xend_%s.npy' % name))
            zend = np.load(join(self.sim_folder, 'zend_%s.npy' % name))
        else:
            xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % name))
            zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % name))
            xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % name))
            zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % name))
            xend = np.load(join(self.sim_folder, 'xend_%s.npy' % name))
            zend = np.load(join(self.sim_folder, 'zend_%s.npy' % name))

        synidx = np.load(join(self.sim_folder, 'synidx_%s_00000.npy' % self.population_sim_name))

        [cell_ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]],
                 lw=1.5, color='0.5', zorder=0, alpha=1)
            for idx in xrange(len(xmid))]
        [cell_ax.plot(xmid[idx], zmid[idx], 'g.', ms=4) for idx in synidx]

        ax_top = fig.add_subplot(3, 3, 2, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_top_n = fig.add_subplot(3, 3, 3, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_mid = fig.add_subplot(3, 3, 5, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_mid_n = fig.add_subplot(3, 3, 6, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_bottom = fig.add_subplot(3, 3, 8, xlim=[1, self.max_freq], xscale=scale, yscale=scale)
        ax_bottom_n = fig.add_subplot(3, 3, 9, xlim=[1, self.max_freq], xscale=scale, yscale=scale)

        all_elec_ax = [ax_top, ax_mid, ax_bottom]
        all_elec_n_ax = [ax_top_n, ax_mid_n, ax_bottom_n]
        cutoff_dist = 1000
        dist_clr = lambda dist: plt.cm.Greys(np.log(dist) / np.log(cutoff_dist))

        for elec in xrange(len(self.elec_z_lateral)):
            if self.elec_x_lateral[elec] > cutoff_dist:
                continue
            clr = dist_clr(self.elec_x_lateral[elec])
            cell_ax.plot(self.elec_x_lateral[elec], self.elec_z_lateral[elec], 'o', c=clr)

            row, col = self._return_elec_row_col(elec)
            all_elec_ax[row].loglog(freqs, sig_psd[elec, :], color=clr, lw=1)
            all_elec_n_ax[row].loglog(freqs, sig_psd[elec, :] / np.max(sig_psd[elec, 1:]), color=clr, lw=1)

        [ax.grid(True) for ax in all_elec_ax + all_elec_n_ax]

        for ax in all_elec_ax + all_elec_n_ax:
            max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()[1:]) for l in ax.get_lines()])))
            ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])

        fig.savefig(join(self.figure_folder, 'lateral_F_%s.png' % self.population_sim_name))


if __name__ == '__main__':
    from param_dicts import shape_function_params
    shape_function_params.update({'mu': -0.5, 'input_region': 'basal',
                                  'distribution': 'linear_decrease',
                                  'cell_number': 0})
    ns = NeuralSimulation(**shape_function_params)
    ns.run_distributed_synaptic_simulation()

