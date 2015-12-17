from __future__ import division
import os
import sys
from os.path import join
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False

from plotting_convention import mark_subplots, simplify_axes
import numpy as np
import pylab as plt
import neuron
import LFPy
import tools


class NeuralSimulation:
    def __init__(self, input_type, **kwargs):

        self.input_type = input_type
        self.param_dict = kwargs
        self._set_input_params()
        self._set_electrode()

        self.cell_name = 'hay'
        self.max_freq = kwargs['max_freq']

        self.holding_potential = kwargs['holding_potential']
        self.conductance_type = kwargs['conductance_type']
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

        self.divide_into_welch = 8.
        self.welch_dict = {'Fs': 1000 / self.timeres_python,
                           'NFFT': int(self.num_tsteps/self.divide_into_welch),
                           'noverlap': int(self.num_tsteps/self.divide_into_welch/2.),
                           'window': plt.window_hanning,
                           'detrend': plt.detrend_mean,
                           'scale_by_freq': True,
                           }
        self.neuron_models = join(self.root_folder, 'neuron_models')

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

    def _set_input_params(self):

        if self.input_type is 'distributed_delta':
            print "Distributed synaptic delta input"
            self._single_neural_sim_function = self._run_distributed_synaptic_simulation
        else:
            raise RuntimeError("Unrecognized input type!")

        self.timeres_NEURON = self.param_dict['timeres_NEURON']
        self.timeres_python = self.param_dict['timeres_python']
        self.cut_off = self.param_dict['cut_off']
        self.end_t = self.param_dict['end_t']
        self.repeats = self.param_dict['repeats']
        self.max_freq = self.param_dict['max_freqs'] if 'max_freqs' in self.param_dict else 500
        self.num_tsteps = round(self.end_t/self.timeres_python + 1)

    def _return_cell(self, mu, distribution):
        if not 'i_QA' in neuron.h.__dict__.keys():
            neuron.load_mechanisms(join(self.neuron_models))

        if self.cell_name == 'hay':
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
                                     'mu_factor': mu,
                                     'g_pas': 0.00005,#0.0002, # / 5,
                                     'distribution': distribution,
                                     'tau_w': 'auto',
                                     #'total_w_conductance': 6.23843378791,# / 5,
                                     'avrg_w_bar': 0.00005 * 2,
                                     'hold_potential': self.holding_potential}]
            }
        else:
            raise ValueError("Unrcognized cell-type")
        cell = LFPy.Cell(**cell_params)
        cell.set_rotation(z=2*np.pi*np.random.random())
        return cell

    def save_neural_sim_single_input_data(self, cell, electrode, input_sec, mu, distribution, cell_number):

        if not self.repeats is None:
            cut_off_idx = (len(cell.tvec) - 1) / self.repeats
            cell.tvec = cell.tvec[-cut_off_idx:] - cell.tvec[-cut_off_idx]
            cell.imem = cell.imem[:, -cut_off_idx:]
            cell.vmem = cell.vmem[:, -cut_off_idx:]

        # tau = '%1.2f' % taum if type(taum) in [float, int] else taum
        # dist_dict = self._get_distribution(dist_dict, cell)

        sim_name = '%s_%s_%s_%s_%+1.1f_%1.5fuS_%05d' % (self.cell_name, self.input_type,
                                                   input_sec, distribution, mu, self.param_dict['syn_weight'], cell_number)

        np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), np.dot(electrode.electrodecoeff, cell.imem))

        if cell_number % 10 == 0:
            np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
            np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)
            np.save(join(self.sim_folder, 'synidx_%s.npy' % sim_name), cell.synidx)

        np.save(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name), electrode.x)
        np.save(join(self.sim_folder, 'elec_y_%s.npy' % self.cell_name), electrode.y)
        np.save(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name), electrode.z)

        np.save(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.xstart)
        np.save(join(self.sim_folder, 'ystart_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.ystart)
        np.save(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.zstart)
        np.save(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.xend)
        np.save(join(self.sim_folder, 'yend_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.yend)
        np.save(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.zend)
        np.save(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.xmid)
        np.save(join(self.sim_folder, 'ymid_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.ymid)
        np.save(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.zmid)
        np.save(join(self.sim_folder, 'diam_%s_%s.npy' % (self.cell_name, self.param_dict['conductance_type'])), cell.diam)

    def _run_distributed_synaptic_simulation(self, mu, input_region, distribution, cell_number):
        plt.seed(123 * cell_number)
        # if os.path.isfile(join(self.sim_folder, 'sig_%s.npy' % sim_name)):
        #     print "Skipping ", mu, input_sec, distribution, tau_w, weight, 'sig_%s.npy' % sim_name
        #     return

        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        cell = self._return_cell(mu, distribution)
        cell, syn, noiseVec = self._make_distributed_synaptic_stimuli(cell, input_region)
        # print "Starting simulation of cell %d" % cell_number

        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        self.save_neural_sim_single_input_data(cell, electrode, input_region, mu, distribution, cell_number)
        self._draw_all_elecs_with_distance(cell, electrode, input_region, mu, distribution, cell_number)
        neuron.h('forall delete_section()')
        del cell, noiseVec, electrode
        for s in syn:
            del d
        del syn

    def _make_distributed_synaptic_stimuli(self, cell, input_sec):

        # Define synapse parameters
        synapse_params = {
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': self.param_dict['syn_tau'],                # syn. time constant
            'weight': self.param_dict['syn_weight'],            # syn. weight
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
        elif input_sec == 'basal':
            input_pos = ['dend']
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

    def _return_elec_subplot_number_with_distance(self, elec_number):
        """ Return the subplot number for the distance study
        """
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        elec_idxs = np.arange(len(self.elec_x)).reshape(num_elec_rows, num_elec_cols)
        row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
        num_plot_cols = num_elec_cols + 3
        plot_number = row * num_plot_cols + col + 4
        return plot_number

    def _return_elec_row_col(self, elec_number):
        """ Return the subplot number for the distance study
        """
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        elec_idxs = np.arange(len(self.elec_x)).reshape(num_elec_rows, num_elec_cols)
        row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
        return row, col

    def plot_cell_to_ax(self, cell, ax):
        [ax.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
                 lw=1.5, color='0.5', zorder=0, alpha=1)
            for idx in xrange(len(cell.xmid))]
        [ax.plot(cell.xmid[idx], cell.zmid[idx], 'g.', ms=4) for idx in cell.synidx]

    def _draw_all_elecs_with_distance(self, cell, electrode, input_region, mu, distribution, cell_number):
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

        sim_name = '%s_%s_%s_%s_%+1.1f_%1.5fuS_%05d' % (self.cell_name, self.input_type,
                                                       input_region, distribution, mu, self.param_dict['syn_weight'], cell_number)
        fig.suptitle(sim_name)
        LFP = 1000 * electrode.LFP#np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))[:, :]

        if self.input_type is 'distributed_delta':
            freqs, sig_psd = tools.return_freq_and_psd_welch(LFP, self.welch_dict)
        else:
            freqs, sig_psd = tools.return_freq_and_psd(self.timeres_python/1000., LFP)

        cutoff_dist = 1000
        dist_clr = lambda dist: plt.cm.Greys(np.log(dist) / np.log(cutoff_dist))

        for elec in xrange(len(self.elec_z)):
            if self.elec_x[elec] > cutoff_dist:
                continue
            clr = dist_clr(self.elec_x[elec])
            cell_ax.plot(self.elec_x[elec], self.elec_z[elec], 'o', c=clr)

            row, col = self._return_elec_row_col(elec)
            all_elec_ax[row].loglog(freqs, sig_psd[elec, :], color=clr, lw=1)
            all_elec_n_ax[row].loglog(freqs, sig_psd[elec, :] / np.max(sig_psd[elec, :]), color=clr, lw=1)

        [ax.grid(True) for ax in all_elec_ax + all_elec_n_ax +
         [ax_i_apic, ax_i_middle, ax_i_soma, ax_v_apic, ax_v_middle, ax_v_soma]]

        fig.savefig(join(self.figure_folder, '%s.png' % sim_name))

class ShapeFunction:
    def __init__(self):
        pass

    def distribute_cellsims_local(self):
        from param_dicts import distributed_delta_params
        ns = NeuralSimulation(**distributed_delta_params)

        secs = ['homogeneous', 'distal_tuft', 'basal']
        mus = [-0.5, 0.0, 2.0]
        dists = ['uniform', 'linear_increase', 'linear_decrease']
        cell_number = 100
        num_sims = len(secs) * len(mus) * len(dists) * cell_number
        i = 0
        for input_sec in secs:
            for mu in mus:
                for channel_dist in dists:
                    for cell_idx in range(cell_number):
                        i += 1
                        print i, '/', num_sims
                        ns._run_distributed_synaptic_simulation(mu, input_sec, channel_dist, cell_idx)

    def distribute_cellsims_MPI_DEPRECATED(self):
        """ Run with
            openmpirun -np 4 python example_mpi.py
        """
        from param_dicts import distributed_delta_params
        ns = NeuralSimulation(**distributed_delta_params)

        from mpi4py import MPI
        COMM = MPI.COMM_WORLD
        SIZE = COMM.Get_size()
        RANK = COMM.Get_rank()

        secs = ['homogeneous', 'distal_tuft', 'basal']
        mus = [-0.5, 0.0, 2.0]
        dists = ['uniform', 'linear_increase', 'linear_decrease']
        cell_number = 100
        num_sims = len(secs) * len(mus) * len(dists) * cell_number

        sim_num = 0
        for input_sec in secs:
            for mu in mus:
                for channel_dist in dists:
                    for cell_idx in range(cell_number):
                        if divmod(sim_num, SIZE)[1] == RANK:
                            print RANK, "simulating ", input_sec, channel_dist, mu, cell_idx
                            ns._run_distributed_synaptic_simulation(mu, input_sec, channel_dist, cell_idx)

                        sim_num += 1


def ShapeFunctionMPI():
    """ Run with
        mpirun -np 4 python example_mpi.py
    """
    from mpi4py import MPI

    class Tags:
        def __init__(self):
            self.READY = 0
            self.DONE = 1
            self.EXIT = 2
            self.START = 3
            self.ERROR = 4
    tags = Tags()
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    num_workers = size - 1

    if size == 1:
        sys.exit()

    if rank == 0:

        secs = ['homogeneous', 'distal_tuft', 'basal']
        mus = [-0.5, 0.0, 2.0]
        dists = ['uniform', 'linear_increase', 'linear_decrease']

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)
        task = 0
        num_cells = 100
        num_tasks = len(secs) * len(mus) * len(dists) * num_cells

        for input_sec in secs:
            for channel_dist in dists:
                for mu in mus:
                    for cell_idx in xrange(num_cells):
                        task += 1
                        sent = False
                        while not sent:
                            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                            source = status.Get_source()
                            tag = status.Get_tag()
                            if tag == tags.READY:
                                comm.send([input_sec, channel_dist, mu, cell_idx], dest=source, tag=tags.START)
                                print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                                sent = True
                            elif tag == tags.DONE:
                                print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                            elif tag == tags.ERROR:
                                print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                                for worker in range(1, num_workers + 1):
                                    comm.send([{}, None], dest=worker, tag=tags.EXIT)
                                sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([{}, None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        from param_dicts import distributed_delta_params
        ns = NeuralSimulation(**distributed_delta_params)
        #print("I am a worker with rank %d." % rank)
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [input_sec, channel_dist, mu, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work on %s %s %+1.1f cell %d\033[0m" % (rank, input_sec,
                                                                                 channel_dist, mu, cell_idx)
                try:
                    ns._run_distributed_synaptic_simulation(mu, input_sec, channel_dist, cell_idx)
                except:
                    print "\033[91mNode %d exiting with ERROR\033[0m" % rank
                    comm.send(None, dest=0, tag=tags.ERROR)
                    sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)

if __name__ == '__main__':

    ShapeFunction()
    # ns = NeuralSimulation(**distributed_delta_params)


