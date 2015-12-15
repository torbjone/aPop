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

class NeuralSimulation:
    def __init__(self, input_type):

        # self._single_neural_sim_function = self._run_distributed_synaptic_simulation
        self.input_type = input_type
        self._set_input_specific_properties()

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

        self._set_electrode()

        self.divide_into_welch = 16.
        self.welch_dict = {'Fs': 1000 / self.timeres_python,
                           'NFFT': int(self.num_tsteps/self.divide_into_welch),
                           'noverlap': int(self.num_tsteps/self.divide_into_welch/2.),
                           'window': plt.window_hanning,
                           'detrend': plt.detrend_mean,
                           'scale_by_freq': True,
                           }
        self.neuron_models = join(self.root_folder, 'neuron_models')
        sys.path.append(join(self.neuron_models, 'hay'))
        from hay_active_declarations import active_declarations
        neuron.load_mechanisms(join(self.neuron_models))

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

    def _set_input_specific_properties(self):

        if self.input_type == 'distributed_delta':
            print "Distributed synaptic delta input"
            self._single_neural_sim_function = self._run_distributed_synaptic_simulation
            self.timeres_NEURON = 2**-4
            self.timeres_python = 2**-4
            self.cut_off = 200
            self.end_t = 10000
            self.repeats = None
            self.max_freq = 500

        else:
            raise ValueError("Unrecognized input_type")

        self.num_tsteps = round(self.end_t/self.timeres_python + 1)



    def _return_cell(self, holding_potential, conductance_type, mu, distribution, tau_w):


        if self.cell_name == 'hay':
            # neuron.load_mechanisms(join(neuron_models, 'hay', 'mod'))

            cell_params = {
                'morphology': join(self.neuron_models, 'hay', 'lfpy_version', 'morphologies', 'cell1.hoc'),
                'v_init': holding_potential,
                'passive': False,           # switch on passive mechs
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
                'timeres_python': self.timeres_python,
                'tstartms': -self.cut_off,          # start time, recorders start at t=0
                'tstopms': self.end_t,
                'custom_code': [join(neuron_models, 'hay', 'lfpy_version', 'custom_codes.hoc')],
                'custom_fun': [hay_active],  # will execute this function
                'custom_fun_args': [{'conductance_type': conductance_type,
                                     'mu_factor': mu,
                                     'g_pas': 0.00005,#0.0002, # / 5,
                                     'distribution': distribution,
                                     'tau_w': tau_w,
                                     #'total_w_conductance': 6.23843378791,# / 5,
                                     'avrg_w_bar': 0.00005 * 2,
                                     'hold_potential': holding_potential}]
            }
        cell = LFPy.Cell(**cell_params)
        return cell


    def save_neural_sim_single_input_data(self, cell, electrode, input_idx,
                             mu, distribution, taum, weight=None):

        if not self.repeats is None:
            cut_off_idx = (len(cell.tvec) - 1) / self.repeats
            cell.tvec = cell.tvec[-cut_off_idx:] - cell.tvec[-cut_off_idx]
            cell.imem = cell.imem[:, -cut_off_idx:]
            cell.vmem = cell.vmem[:, -cut_off_idx:]

        tau = '%1.2f' % taum if type(taum) in [float, int] else taum
        # dist_dict = self._get_distribution(dist_dict, cell)

        sim_name = '%s_%s_%+1.1f_%s_%dms_%1.5fuS' % (self.cell_name, self.input_type, mu, distribution, tau, weight)


        np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
        # np.save(join(self.sim_folder, 'dist_dict_%s.npy' % sim_name), dist_dict)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), np.dot(electrode.electrodecoeff, cell.imem))

        np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)
        np.save(join(self.sim_folder, 'synidx_%s.npy' % sim_name), cell.synidx)

        np.save(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name), electrode.x)
        np.save(join(self.sim_folder, 'elec_y_%s.npy' % self.cell_name), electrode.y)
        np.save(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name), electrode.z)

        np.save(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xstart)
        np.save(join(self.sim_folder, 'ystart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ystart)
        np.save(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zstart)
        np.save(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xend)
        np.save(join(self.sim_folder, 'yend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.yend)
        np.save(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zend)
        np.save(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xmid)
        np.save(join(self.sim_folder, 'ymid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ymid)
        np.save(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zmid)
        np.save(join(self.sim_folder, 'diam_%s_%s.npy' % (self.cell_name, self.conductance)), cell.diam)



    def _run_distributed_synaptic_simulation(self, mu, input_sec, distribution, tau_w, weight):
        plt.seed(1234)


        # if os.path.isfile(join(self.sim_folder, 'sig_%s.npy' % sim_name)):
        #     print "Skipping ", mu, input_sec, distribution, tau_w, weight, 'sig_%s.npy' % sim_name
        #     return

        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        cell = self._return_cell(mu, distribution, tau_w)
        cell, syn, noiseVec = self._make_distributed_synaptic_stimuli_equal_density(cell, input_sec, weight)
        print "Starting simulation ..."

        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        self.save_neural_sim_single_input_data(cell, electrode, input_sec, mu, distribution, tau_w, weight)
        neuron.h('forall delete_section()')
        del cell, syn, noiseVec, electrode

    def _make_distributed_synaptic_stimuli(self, cell, input_sec, weight=0.001, **kwargs):

        # Define synapse parameters
        synapse_params = {
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': .1,                # syn. time constant
            'weight': weight,            # syn. weight
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

    ns = NeuralSimulation("distributed_delta")


