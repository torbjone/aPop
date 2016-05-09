from __future__ import division
import param_dicts
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False
import pylab as plt
import numpy as np
import os
from os.path import join
import LFPy
import sys
from NeuralSimulation import NeuralSimulation
import tools
import scipy.fftpack as sf

def return_elec_row_col(elec_number, elec_x, elec_z):
    """ Return the subplot number for the distance study
    """
    num_elec_cols = len(set(elec_x))
    num_elec_rows = len(set(elec_z))
    elec_idxs = np.arange(len(elec_x)).reshape(num_elec_rows, num_elec_cols)
    row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
    return row, col

def plot_population_LFP(param_dict):

    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')

    for input_region in param_dict['input_regions']:
        for distribution in param_dict['distributions']:
            for correlation in param_dict['correlations']:
                param_dict.update({'input_region': input_region,
                               'cell_number': 0,
                               'distribution': distribution,
                               'correlation': correlation,
                               })

                param_dict['mu'] = 0.0
                ns = NeuralSimulation(**param_dict)
                name = 'summed_signal_%s_350um' % ns.population_sim_name
                # coherence = np.load(join(ns.sim_folder, 'trick_coherence_%s.npy' % ns.population_sim_name))
                # coherence_freq = np.load(join(ns.sim_folder, 'trick_coherence_freq_%s.npy' % ns.population_sim_name))


                sim_folder = join(param_dict['root_folder'], 'shape_function', 'simulations')
                F_name = 'shape_function_hay_generic_%s_%1.1f_%s_0.00' % (distribution, 0.0, input_region)
                F = np.load(join(sim_folder, 'F_%s.npy' % F_name))
                F_freq = np.load(join(sim_folder, 'welch_freqs.npy'))

                # print coherence.shape, F_freq.shape, coherence_freq.shape
                # print F_freq[1] - F_freq[0], coherence_freq[1] - coherence_freq[0]

                # print F_freq, coherence_freq

                # bin_psd = np.zeros(len(mlab_freqs))
                # df = mlab_freqs[1] - mlab_freqs[0]
                # print name
                # for m, mfreq in enumerate(mlab_freqs):
                    # print mfreq - df/2, mfreq, mfreq + df/2, freqs[(freqs >= mfreq - df/2) & (freqs < mfreq + df/2)]
                    # bin_psd[m] = np.average(ff_psd[(freqs >= mfreq - df/2) & (freqs < mfreq + df/2)])


                param_dict['mu'] = -0.5
                ns = NeuralSimulation(**param_dict)
                name_reg = 'summed_signal_%s_350um' % ns.population_sim_name

                param_dict['mu'] = 2.0
                ns = NeuralSimulation(**param_dict)
                name_res = 'summed_signal_%s_350um' % ns.population_sim_name
                # print name_res

                xmid = np.load(join(folder, 'xmid_hay_generic.npy'))
                zmid = np.load(join(folder, 'zmid_hay_generic.npy'))
                xstart = np.load(join(folder, 'xstart_hay_generic.npy'))
                zstart = np.load(join(folder, 'zstart_hay_generic.npy'))
                xend = np.load(join(folder, 'xend_hay_generic.npy'))
                zend = np.load(join(folder, 'zend_hay_generic.npy'))

                synidx = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))

                lfp = np.load(join(folder, '%s.npy' % name))
                lfp_reg = np.load(join(folder, '%s.npy' % name_reg))
                lfp_res = np.load(join(folder, '%s.npy' % name_res))

                # print lfp - lfp_res
                # print lfp_res

                elec_x = param_dict['electrode_parameters']['x']
                elec_z = param_dict['electrode_parameters']['z']
                # print elec_x.reshape(3, 30)
                # print elec_z.reshape(3, 30)
                plt.close('all')
                fig = plt.figure(figsize=(18, 10))
                fig.suptitle(name)
                ax_morph = fig.add_subplot(1, 6, 1, aspect=1, frameon=False, xticks=[])
                [ax_morph.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1.5, color='0.5', zorder=0, alpha=1)
                    for idx in xrange(len(xmid))]
                ax_morph.plot(xmid[synidx], zmid[synidx], '.', c='green', ms=4)
                fig.subplots_adjust(right=0.99, wspace=0.5, hspace=0.3)
                num_elec_cols = 30
                col_cut_off = 20

                for elec in range(lfp.shape[0]):
                    row, col = return_elec_row_col(elec, elec_x, elec_z)
                    if col % 4 != 0 or col > col_cut_off:
                        continue
                    num_plot_cols = col_cut_off/4 + 2
                    plot_number = row * num_plot_cols + col/4 + 2
                    ax = fig.add_subplot(3, num_plot_cols, plot_number, aspect=0.5, ylim=[1e-9, 1e-0], xlim=[1e0, 5e2],
                                         title='%1.1f $\mu$m' % (elec_x[elec]))
                    freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                    freq, psd_reg = tools.return_freq_and_psd(ns.timeres_python/1000.,  lfp_reg[elec])
                    freq, psd_res = tools.return_freq_and_psd(ns.timeres_python/1000., lfp_res[elec])
                    ax.loglog(freq, psd[0], 'k')
                    ax.loglog(freq, psd_reg[0], 'r')
                    ax.loglog(freq, psd_res[0], 'b')

                plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], '%s.png' % name))
                plt.close('all')

def initialize_population(param_dict):
    print "Initializing cell positions and rotations ..."
    x_y_z_rot = np.zeros((param_dict['num_cells'], 4))
    for cell_number in range(param_dict['num_cells']):
        plt.seed(123 * cell_number)
        rotation = 2*np.pi*np.random.random()
        z = param_dict['layer_5_thickness'] * (np.random.random() - 0.5)
        x, y = 2 * (np.random.random(2) - 0.5) * param_dict['population_radius']
        while np.sqrt(x**2 + y**2) > param_dict['population_radius']:
            x, y = 2 * (np.random.random(2) - 0.5) * param_dict['population_radius']
        x_y_z_rot[cell_number, :] = [x, y, z, rotation]
    r = np.array([np.sqrt(d[0]**2 + d[1]**2) for d in x_y_z_rot])
    argsort = np.argsort(r)
    x_y_z_rot = x_y_z_rot[argsort]

    np.save(join(param_dict['root_folder'], param_dict['save_folder'],
                 'x_y_z_rot_%s.npy' % param_dict['name']), x_y_z_rot)

    print "Initializing input spike trains"
    plt.seed(123)
    num_synapses = 1000
    firing_rate = 5
    num_trains = int(num_synapses/0.01)
    all_spiketimes = {}
    input_method = LFPy.inputgenerators.stationary_poisson
    for idx in xrange(num_trains):
        all_spiketimes[idx] = input_method(1, firing_rate, -param_dict['cut_off'], param_dict['end_t'])[0]
    np.save(join(param_dict['root_folder'], param_dict['save_folder'],
                         'all_spike_trains_%s.npy' % param_dict['name']), all_spiketimes)


def plot_population(param_dict):
    print "Plotting population"
    x_y_z_rot = np.load(join(param_dict['root_folder'], param_dict['save_folder'],
                         'x_y_z_rot_%s.npy' % param_dict['name']))
    all_spiketimes = np.load(join(param_dict['root_folder'], param_dict['save_folder'],
                         'all_spike_trains_%s.npy' % param_dict['name'])).item()

    plt.subplot(221, xlabel='x', ylabel='y', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 1], edgecolor='none', s=2)

    plt.subplot(222, xlabel='x', ylabel='z', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 2], edgecolor='none', s=2)

    plt.subplot(212, ylabel='Cell number', xlabel='Time', title='Raster plot', frameon=False)
    for idx in range(200):
        plt.scatter(all_spiketimes[idx], np.ones(len(all_spiketimes[idx])) * idx, edgecolors='none', s=4)

    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'],
                     'population_%s.png' % param_dict['name']))


def sum_one_population(param_dict, num_cells, num_tsteps):

    x_y_z_rot = np.load(os.path.join(param_dict['root_folder'], param_dict['save_folder'],
                             'x_y_z_rot_%s.npy' % param_dict['name']))
    summed_sig = np.zeros((len(param_dict['electrode_parameters']['x']), num_tsteps))
    # Cells are numbered with increasing distance from population center. We exploit this
    subpopulation_idx = 0
    ns = None
    pop_radius = 0

    sample_freq = sf.fftfreq(num_tsteps, d=param_dict['timeres_python'] / 1000.)
    pidxs = np.where(sample_freq >= 0)
    # freqs = sample_freq[pidxs]
    xfft_norm_sum = np.zeros((len(param_dict['electrode_parameters']['x']), len(pidxs[0])), dtype=complex)

    for cell_number in xrange(num_cells):
        param_dict.update({'cell_number': cell_number})

        r = np.sqrt(x_y_z_rot[cell_number, 0]**2 + x_y_z_rot[cell_number, 1]**2)
        pop_radius = param_dict['population_radii'][subpopulation_idx]

        ns = NeuralSimulation(**param_dict)
        sim_name = ns.sim_name
        lfp = 1000 * np.load(join(ns.sim_folder, 'sig_%s.npy' % sim_name))  # uV
        s = sf.fft(lfp, axis=1)[:, pidxs[0]]
        s_norm = s / np.abs(s)

        # freqs, lfp_psd = tools.return_freq_and_psd_welch(lfp[:, :], ns.welch_dict)
        if not summed_sig.shape == lfp.shape:
            print "Reshaping LFP time signal", lfp.shape
            summed_sig = np.zeros(lfp.shape)

        if not xfft_norm_sum.shape == s_norm.shape:
            print "Reshaping coherence signal", xfft_norm_sum.shape, s_norm.shape
            xfft_norm_sum = np.zeros(s_norm.shape)

        xfft_norm_sum[:] += s_norm[:]

        if r > pop_radius:
            print "Saving population of radius ", pop_radius
            print "r(-1): ", np.sqrt(x_y_z_rot[cell_number - 1, 0]**2 + x_y_z_rot[cell_number - 1, 1]**2)
            print "r: ", r
            np.save(join(ns.sim_folder, 'summed_signal_%s_%dum.npy' %
                         (ns.population_sim_name, pop_radius)), summed_sig)
            subpopulation_idx += 1
        summed_sig += lfp
    print "Saving population of radius ", pop_radius
    np.save(join(ns.sim_folder, 'summed_signal_%s_%dum.npy' %
                 (ns.population_sim_name, pop_radius)), summed_sig)

    c_phi = (np.abs(xfft_norm_sum) ** 2 - num_cells) / (num_cells * (num_cells - 1))
    np.save(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name), c_phi)
    # plot_summed_signal(summed_sig)


def sum_population_generic(param_dict):

    x_y_z_rot = np.load(os.path.join(param_dict['root_folder'], param_dict['save_folder'],
                         'x_y_z_rot_%s.npy' % param_dict['name']))
    # print param_dict['population_radii']
    num_cells = 2000
    num_tsteps = round(param_dict['end_t']/param_dict['timeres_python'] + 1)
    for input_region in param_dict['input_regions']:
        param_dict.update({'input_region': input_region})
        for distribution in param_dict['distributions']:
            param_dict.update({'distribution': distribution})
            for mu in param_dict['mus']:
                param_dict.update({'mu': mu})
                for correlation in param_dict['correlations']:
                    param_dict.update({'correlation': correlation})
                    print input_region, distribution, mu, correlation
                    sum_one_population(param_dict, num_cells, x_y_z_rot, num_tsteps)


def sum_population_mpi(param_dict):
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
    num_cells = 2000
    num_tsteps = round(param_dict['end_t']/param_dict['timeres_python'] + 1)

    if size == 1:
        print "Can't do MPI with one core!"
        sys.exit()

    if rank == 0:

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)

        task = 0
        num_tasks = (len(param_dict['input_regions']) * len(param_dict['mus']) *
                     len(param_dict['distributions']) * len(param_dict['correlations']))

        # print param_dict['population_radii']

        for input_region in param_dict['input_regions']:
            param_dict.update({'input_region': input_region})
            for distribution in param_dict['distributions']:
                param_dict.update({'distribution': distribution})
                for mu in param_dict['mus']:
                    param_dict.update({'mu': mu})
                    for correlation in param_dict['correlations']:
                        param_dict.update({'correlation': correlation})
                        print input_region, distribution, mu, correlation
                        task += 1
                        sent = False
                        while not sent:
                            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                            source = status.Get_source()
                            tag = status.Get_tag()
                            if tag == tags.READY:
                                comm.send([param_dict], dest=source, tag=tags.START)
                                print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                                sent = True
                            elif tag == tags.DONE:
                                print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                            elif tag == tags.ERROR:
                                print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                                for worker in range(1, num_workers + 1):
                                    comm.send([None], dest=worker, tag=tags.EXIT)
                                sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [param_dict] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                try:
                    sum_one_population(param_dict, num_cells, num_tsteps)
                except:
                    print "\033[91mNode %d exiting with ERROR\033[0m" % rank
                    comm.send(None, dest=0, tag=tags.ERROR)
                    sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)

def PopulationMPI():
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
        print "Can't do MPI with one core!"
        sys.exit()

    if rank == 0:

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)
        task = 0
        num_cells = 2000
        num_tasks = (len(param_dict['input_regions']) * len(param_dict['mus']) *
                     len(param_dict['distributions']) * len(param_dict['correlations']) * (num_cells))

        for input_region in param_dict['input_regions']:
            for distribution in param_dict['distributions']:
                for mu in param_dict['mus']:
                    for correlation in param_dict['correlations']:
                        for cell_idx in xrange(0, num_cells):
                            task += 1
                            sent = False
                            while not sent:
                                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                                source = status.Get_source()
                                tag = status.Get_tag()
                                if tag == tags.READY:
                                    comm.send([input_region, distribution, mu, correlation, cell_idx], dest=source, tag=tags.START)
                                    print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                                    sent = True
                                elif tag == tags.DONE:
                                    print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                                elif tag == tags.ERROR:
                                    print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                                    for worker in range(1, num_workers + 1):
                                        comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
                                    sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [input_region, distribution, mu, correlation, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work on %s %s %+1.1f %1.2f cell %d\033[0m" % (rank, input_region,
                                                                                distribution, mu, correlation, cell_idx)
                try:
                    os.system("python %s %s %1.2f %d %s %1.1f" % (sys.argv[0], input_region, correlation, cell_idx,
                                                                  distribution, mu))
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
    conductance = 'generic'
    if conductance is 'generic':
        from param_dicts import generic_population_params as param_dict
    else:
        from param_dicts import classic_population_params as param_dict

    # plot_population_LFP(param_dict)
    # sys.exit()

    if len(sys.argv) >= 3:
        cell_number = int(sys.argv[3])
        input_region = sys.argv[1]
        correlation = float(sys.argv[2])
        if conductance is 'generic':
            distribution = sys.argv[4]
            mu = float(sys.argv[5])
            param_dict.update({'input_region': input_region,
                               'cell_number': cell_number,
                               'mu': mu,
                               'distribution': distribution,
                               'correlation': correlation,
                               })
        else:
            conductance_type = sys.argv[4]
            holding_potential = int(sys.argv[5])
            param_dict.update({'input_region': input_region,
                               'cell_number': cell_number,
                               'conductance_type': conductance_type,
                               'holding_potential': holding_potential,
                               'correlation': correlation,
                               })
        ns = NeuralSimulation(**param_dict)
        ns.run_distributed_synaptic_simulation()
    elif len(sys.argv) == 2 and sys.argv[1] == 'initialize':
        initialize_population(param_dict)
        plot_population(param_dict)
    elif len(sys.argv) == 2 and sys.argv[1] == 'MPI':
        PopulationMPI()
    elif len(sys.argv) == 2 and sys.argv[1] == 'sum':
        sum_population_mpi(param_dict)
        # sum_population_generic(param_dict)
