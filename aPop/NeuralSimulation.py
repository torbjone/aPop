import os
import sys
from os.path import join

if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False

import numpy as np
import pylab as plt
import neuron
import LFPy


# TODO: 1) give cell number as input
# TODO: 2) plot example cell time series
# TODO: 3) Only save imem and vmem from small fraction of cells

class NeuralSimulation:
    def __init__(self, input_type, **kwargs):

        self.input_type = input_type
        self._set_input_params(kwargs)
        self._set_electrode()

        self.cell_name = 'hay'

        self.username = os.getenv('USER')

        if at_stallo:
            self.root_folder = join('/global', 'work', self.username, 'aPop')
            self.figure_folder = join(self.root_folder, 'generic_study')
            self.sim_folder = join(self.root_folder, 'generic_study', self.cell_name)
        else:
            self.root_folder = join('/home', self.username, 'work', 'aPop')
            self.figure_folder = join(self.root_folder, 'generic_study')
            self.sim_folder = join(self.root_folder, 'generic_study', self.cell_name)
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
        if not os.path.isdir(self.sim_folder):
            os.mkdir(self.sim_folder)

        self.divide_into_welch = 16.
        self.welch_dict = {'Fs': 1000 / self.timeres_python,
                           'NFFT': int(self.num_tsteps/self.divide_into_welch),
                           'noverlap': int(self.num_tsteps/self.divide_into_welch/2.),
                           'window': plt.window_hanning,
                           'detrend': plt.detrend_mean,
                           'scale_by_freq': True,
                           }
        self.neuron_models = join(self.root_folder, 'neuron_models')
        self._run_distributed_synaptic_simulation(0, 'homogeneous', 'uniform', kwargs)


    def _set_electrode(self):
        self.distances = 10 * np.exp(np.linspace(0, np.log(1000), 30))
        elec_x, elec_z = np.meshgrid(self.distances, np.array([1000, 500, 0]))
        self.elec_x = elec_x.flatten()
        self.elec_z = elec_z.flatten()
        self.elec_y = np.zeros(len(self.elec_z))
        self.electrode_parameters = {
                'sigma': 0.3,
                'x': self.elec_x,
                'y': self.elec_y,
                'z': self.elec_z
        }

    def _set_input_params(self, kwargs):

        if self.input_type is 'distributed_delta':
            print "Distributed synaptic delta input"
            self._single_neural_sim_function = self._run_distributed_synaptic_simulation
        else:
            raise RuntimeError("Unrecognized input type!")

        self.timeres_NEURON = kwargs['timeres_NEURON']
        self.timeres_python = kwargs['timeres_python']
        self.cut_off = kwargs['cut_off']
        self.end_t = kwargs['end_t']
        self.repeats = kwargs['repeats']
        self.max_freq = kwargs['max_freqs'] if 'max_freqs' in kwargs else 500
        self.num_tsteps = round(self.end_t/self.timeres_python + 1)

    def _return_cell(self, mu, distribution, kwargs):
        if not 'i_QA' in neuron.h.__dict__.keys():
            neuron.load_mechanisms(join(self.neuron_models))

        if self.cell_name == 'hay':
            sys.path.append(join(self.neuron_models, 'hay'))
            from hay_active_declarations import active_declarations
            cell_params = {
                'morphology': join(self.neuron_models, 'hay', 'cell1.hoc'),
                'v_init': kwargs['holding_potential'],
                'passive': False,           # switch on passive mechs
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
                'timeres_python': self.timeres_python,
                'tstartms': -self.cut_off,          # start time, recorders start at t=0
                'tstopms': self.end_t,
                'custom_code': [join(self.neuron_models, 'hay', 'custom_codes.hoc')],
                'custom_fun': [active_declarations],  # will execute this function
                'custom_fun_args': [{'conductance_type': kwargs['conductance_type'],
                                     'mu_factor': mu,
                                     'g_pas': 0.00005,#0.0002, # / 5,
                                     'distribution': distribution,
                                     'tau_w': 'auto',
                                     #'total_w_conductance': 6.23843378791,# / 5,
                                     'avrg_w_bar': 0.00005 * 2,
                                     'hold_potential': kwargs['holding_potential']}]
            }
        cell = LFPy.Cell(**cell_params)
        return cell


    def save_neural_sim_single_input_data(self, cell, electrode, input_sec, mu, distribution, kwargs):

        if not self.repeats is None:
            cut_off_idx = (len(cell.tvec) - 1) / self.repeats
            cell.tvec = cell.tvec[-cut_off_idx:] - cell.tvec[-cut_off_idx]
            cell.imem = cell.imem[:, -cut_off_idx:]
            cell.vmem = cell.vmem[:, -cut_off_idx:]

        # tau = '%1.2f' % taum if type(taum) in [float, int] else taum
        # dist_dict = self._get_distribution(dist_dict, cell)

        sim_name = '%s_%s_%s_%s_%+1.1f_%1.5fuS' % (self.cell_name, self.input_type,
                                                   input_sec, distribution, mu, kwargs['syn_weight'])

        np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
        # np.save(join(self.sim_folder, 'dist_dict_%s.npy' % sim_name), dist_dict)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), np.dot(electrode.electrodecoeff, cell.imem))

        np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)
        np.save(join(self.sim_folder, 'synidx_%s.npy' % sim_name), cell.synidx)

        np.save(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name), electrode.x)
        np.save(join(self.sim_folder, 'elec_y_%s.npy' % self.cell_name), electrode.y)
        np.save(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name), electrode.z)

        np.save(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.xstart)
        np.save(join(self.sim_folder, 'ystart_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.ystart)
        np.save(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.zstart)
        np.save(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.xend)
        np.save(join(self.sim_folder, 'yend_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.yend)
        np.save(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.zend)
        np.save(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.xmid)
        np.save(join(self.sim_folder, 'ymid_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.ymid)
        np.save(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.zmid)
        np.save(join(self.sim_folder, 'diam_%s_%s.npy' % (self.cell_name, kwargs['conductance_type'])), cell.diam)



    def _run_distributed_synaptic_simulation(self, mu, input_sec, distribution, kwargs):
        plt.seed(1234)
        # if os.path.isfile(join(self.sim_folder, 'sig_%s.npy' % sim_name)):
        #     print "Skipping ", mu, input_sec, distribution, tau_w, weight, 'sig_%s.npy' % sim_name
        #     return

        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        cell = self._return_cell(mu, distribution, kwargs)
        cell, syn, noiseVec = self._make_distributed_synaptic_stimuli(cell, input_sec, kwargs)
        print "Starting simulation ..."

        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        self.save_neural_sim_single_input_data(cell, electrode, input_sec, mu, distribution, kwargs)
        neuron.h('forall delete_section()')
        del cell, syn, noiseVec, electrode

    def _make_distributed_synaptic_stimuli(self, cell, input_sec, kwargs):

        # Define synapse parameters
        synapse_params = {
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': kwargs['syn_tau'],                # syn. time constant
            'weight': kwargs['syn_weight'],            # syn. weight
            'record_current': False,
        }
        if input_sec == 'distal_tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 900
        elif input_sec == 'tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 600
        elif input_sec == 'homogeneous':
            input_pos = ['apic', 'dend']
            maxpos = 10000
            minpos = -10000
        else:
            # input_pos = input_sec
            # maxpos = 10000
            # minpos = -10000
            raise RuntimeError("Use other input section")
        num_synapses = 1000
        cell_input_idxs = cell.get_rand_idx_area_norm(section=input_pos, nidx=num_synapses,
                                                      z_min=minpos, z_max=maxpos)
        spike_trains = LFPy.inputgenerators.stationary_poisson(num_synapses, 5, cell.tstartms, cell.tstopms)
        synapses = self.set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)
        #self.input_plot(cell, cell_input_idxs, spike_trains)

        return cell, synapses, None


    def set_input_spiketrain(self, cell, cell_input_idxs, spike_trains, synapse_params):
        synapse_list = []
        for number, comp_idx in enumerate(cell_input_idxs):
            synapse_params.update({'idx': int(comp_idx)})
            s = LFPy.Synapse(cell, **synapse_params)
            s.set_spike_times(spike_trains[number])
            synapse_list.append(s)
        return synapse_list


if __name__ == '__main__':
    from param_dicts import distributed_delta_params

    ns = NeuralSimulation(**distributed_delta_params)


