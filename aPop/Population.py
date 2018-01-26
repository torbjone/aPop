
print("HERE")

import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False
from plotting_convention import *
import pylab as plt
import numpy as np
import os
import sys
from os.path import join
# from suppress_print import suppress_stdout_stderr
# with suppress_stdout_stderr():
import LFPy
from NeuralSimulation import NeuralSimulation


def initialize_population(param_dict):

    print("Initializing cell positions and rotations ...")
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

    if not os.path.isdir(join(param_dict['root_folder'], param_dict['save_folder'])):
        os.mkdir(join(param_dict['root_folder'], param_dict['save_folder']))

    np.save(join(param_dict['root_folder'], param_dict['save_folder'],
                 'x_y_z_rot_%s.npy' % param_dict['name']), x_y_z_rot)

    print("Initializing input spike trains")
    plt.seed(123)
    num_synapses = param_dict['num_synapses']
    firing_rate = param_dict['input_firing_rate']
    num_trains = int(num_synapses/0.01)
    all_spiketimes = {}
    # input_method = LFPy.inputgenerators.stationary_poisson
    input_method = LFPy.inputgenerators.get_activation_times_from_distribution

    for idx in range(num_trains):
        # all_spiketimes[idx] = input_method(1, firing_rate, -param_dict['cut_off'], param_dict['end_t'])[0]
        all_spiketimes[idx] = input_method(1, -param_dict['cut_off'],
                                   param_dict['end_t'],
                                   rvs_args=dict(loc=0., scale=1000. / firing_rate))[0]
    np.save(join(param_dict['root_folder'], param_dict['save_folder'],
                         'all_spike_trains_%s.npy' % param_dict['name']), all_spiketimes)


def plot_population(param_dict):
    print("Plotting population")
    x_y_z_rot = np.load(join(param_dict['root_folder'], param_dict['save_folder'],
                         'x_y_z_rot_%s.npy' % param_dict['name']))

    plt.subplot(221, xlabel='x', ylabel='y', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 1], edgecolor='none', s=2, color='k')

    plt.subplot(222, xlabel='x', ylabel='z', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 2], edgecolor='none', s=2, color='k')

    plt.subplot(212, ylabel='Cell number', xlabel='Time', title='Raster plot', frameon=False)
    all_spiketimes = np.load(join(param_dict['root_folder'], param_dict['save_folder'],
                         'all_spike_trains_%s.npy' % param_dict['name'])).item()
    for idx in range(200):
        plt.scatter(all_spiketimes[idx], np.ones(len(all_spiketimes[idx])) * idx, color='r',
                    edgecolors='none', s=4)

    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'],
                     'population_%s.png' % param_dict['name']))


def sum_and_remove(param_dict, num_cells, remove=False):
    print("summing LFP and removing single LFP files")
    import time
    num_tsteps = int(round(param_dict['end_t']/param_dict['dt'] + 1))

    # param_dict.update({'cell_number': 0})
    # filename = join(ns.sim_folder, 'summed_center_signal_%s_%dum_.npy' %
    #              (ns.population_sim_name, 637))
    # if os.path.isfile(filename):
    #     return
    x_y_z_rot = np.load(os.path.join(param_dict['root_folder'], param_dict['save_folder'],
                             'x_y_z_rot_%s.npy' % param_dict['name']))
    summed_center_sig = np.zeros((len(param_dict['center_electrode_parameters']['x']), num_tsteps))
    summed_center_sig_half_density = np.zeros((len(param_dict['center_electrode_parameters']['x']), num_tsteps))

    ns = None
    r = None
    for cell_number in range(0, num_cells):
        param_dict.update({'cell_number': cell_number})
        r = np.sqrt(x_y_z_rot[cell_number, 0]**2 + x_y_z_rot[cell_number, 1]**2)
        ns = NeuralSimulation(**param_dict)
        sim_name = ns.sim_name

        loaded = False
        t0 = time.time()
        while not loaded:
            try:
                center_lfp = 1000 * np.load(join(ns.sim_folder, 'center_sig_%s.npy' % sim_name))  # uV
                loaded = True
            except IOError:
                print("Could not load ", cell_number, join(ns.sim_folder, 'center_sig_%s.npy' % sim_name), time.time() - t0)
                time.sleep(5)
            if time.time() - t0 > 60 * 60:
                print("waited 60 minute for %s. Can wait no longer" % sim_name)
                return False

        # if not summed_center_sig.shape == center_lfp.shape:
        #     print "Reshaping LFP time signal", center_lfp.shape
        #     summed_center_sig = np.zeros(center_lfp.shape)
        summed_center_sig += center_lfp

        if not cell_number % 2:
            summed_center_sig_half_density += center_lfp

        if cell_number in [10, 50, 100, 200, 500, 1000, 4000]:
            np.save(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                         (ns.population_sim_name, r)), summed_center_sig)
            np.save(join(ns.sim_folder, 'summed_center_signal_half_density_%s_%dum.npy' %
                         (ns.population_sim_name, r)), summed_center_sig_half_density)

    np.save(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                 (ns.population_sim_name, r)), summed_center_sig)
    np.save(join(ns.sim_folder, 'summed_center_signal_half_density_%s_%dum.npy' %
                 (ns.population_sim_name, r)), summed_center_sig_half_density)

    if remove:
        param_dict.update({'cell_number': 0})
        ns = NeuralSimulation(**param_dict)
        sig_name = join(ns.sim_folder, 'center_sig_%s_*.npy' % ns.population_sim_name)
        print("Remove files %s" % sig_name)
        os.system("rm %s" % sig_name)
    return True


def PopulationMPIgeneric(param_dict):
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
        print("Can't do MPI with one core!")
        sys.exit()

    if rank == 0:

        print(("\033[95m Master starting with %d workers\033[0m" % num_workers))
        task = 0
        num_cells = 1000 if at_stallo else 2
        num_tasks = (len(param_dict['input_regions']) * len(param_dict['mus']) *
                     len(param_dict['distributions']) * len(param_dict['correlations']) * (num_cells))
        for input_region in param_dict['input_regions']:
            param_dict['input_region'] = input_region
            for distribution in param_dict['distributions']:
                param_dict['distribution'] = distribution
                for mu in param_dict['mus']:
                    param_dict['mu'] = mu
                    for correlation in param_dict['correlations']:
                        param_dict["correlation"] = correlation
                        param_dict.update({'cell_number': 0})
                        ns = NeuralSimulation(**param_dict)

                        if os.path.isfile(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                            (ns.population_sim_name, 999))):
                            print("SKIPPING POPULATION ", ns.population_sim_name)
                            continue

                        for cell_idx in range(0, num_cells):
                            task += 1
                            sent = False
                            while not sent:
                                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                                source = status.Get_source()
                                tag = status.Get_tag()
                                if tag == tags.READY:
                                    comm.send([input_region, distribution, mu, correlation, cell_idx], dest=source, tag=tags.START)
                                    # print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                                    sent = True
                                elif tag == tags.DONE:
                                    pass
                                    # print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                                elif tag == tags.ERROR:
                                    print("\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source)
                                    for worker in range(1, num_workers + 1):
                                        comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
                                    sys.exit()
                        success = sum_and_remove(param_dict, num_cells, True)
                        if not success:
                            print("Failed to sum. Exiting")
                            for worker in range(1, num_workers + 1):
                                comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
                                sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        import time
        while True:
            # if rank % 2 == 0:
            #     print "Rank %d sleeping" % rank
                # time.sleep(60)
            # else:
            comm.send(None, dest=0, tag=tags.READY)
            [input_region, distribution, mu, correlation, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print("\033[93m%d put to work on %s %s %s %1.2f cell %d\033[0m" % (rank, input_region,
                                                                                distribution, str(mu), correlation, cell_idx))
                try:
                    os.system("python %s %s %1.2f %d %s %s" % (sys.argv[0], input_region, correlation, cell_idx,
                                                                  distribution, str(mu)))
                except:
                    print("\033[91mNode %d exiting with ERROR\033[0m" % rank)
                    comm.send(None, dest=0, tag=tags.ERROR)
                    sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print("\033[93m%d exiting\033[0m" % rank)
                break
        comm.send(None, dest=0, tag=tags.EXIT)



def PopulationMPIclassic(param_dict):
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
        print("Can't do MPI with one core!")
        sys.exit()

    if rank == 0:

        print(("\033[95m Master starting with %d workers\033[0m" % num_workers))
        task = 0
        num_cells = 10000 if at_stallo else 2
        num_tasks = (len(param_dict['input_regions']) * len(param_dict['holding_potentials']) *
                     len(param_dict['conductance_types']) * len(param_dict['correlations']) * (num_cells))

        for holding_potential in param_dict['holding_potentials']:
            param_dict['holding_potential'] = holding_potential
            for input_region in param_dict['input_regions']:
                param_dict['input_region'] = input_region
                for conductance_type in param_dict['conductance_types']:
                    param_dict['conductance_type'] = conductance_type
                    for correlation in param_dict['correlations']:
                        param_dict["correlation"] = correlation
                        param_dict.update({'cell_number': 0})
                        ns = NeuralSimulation(**param_dict)
                        if os.path.isfile(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                            (ns.population_sim_name, 999))):
                            print("SKIPPING POPULATION ", ns.population_sim_name)
                            continue

                        for cell_idx in range(0, num_cells):
                            task += 1
                            sent = False
                            while not sent:
                                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                                source = status.Get_source()
                                tag = status.Get_tag()
                                if tag == tags.READY:
                                    comm.send([input_region, conductance_type, holding_potential, correlation, cell_idx], dest=source, tag=tags.START)
                                    print("\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source))
                                    sent = True
                                elif tag == tags.DONE:
                                    print("\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks))
                                elif tag == tags.ERROR:
                                    print("\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source)
                                    for worker in range(1, num_workers + 1):
                                        comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
                                    sys.exit()
                        success = sum_and_remove(param_dict, num_cells, True)
                        if not success:
                            print("Failed to sum. Exiting")
                            for worker in range(1, num_workers + 1):
                                comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
                                sys.exit()
        for worker in range(1, num_workers + 1):
            comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        # import time
        while True:
        #     if rank % 2 == 0:
        #         print "Rank %d sleeping" % rank
        #         time.sleep(60)
        #     else:
            comm.send(None, dest=0, tag=tags.READY)
            [input_region, conductance_type, holding_potential, correlation, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print("\033[93m%d put to work on %s %s %s %1.2f cell %d\033[0m" % (rank, input_region,
                                                                                conductance_type, str(holding_potential), correlation, cell_idx))
                try:
                    os.system("python %s %s %1.2f %d %s %s" % (sys.argv[0], input_region, correlation, cell_idx,
                                                                  conductance_type, str(holding_potential)))
                except:
                    print("\033[91mNode %d exiting with ERROR\033[0m" % rank)
                    comm.send(None, dest=0, tag=tags.ERROR)
                    sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print("\033[93m%d exiting\033[0m" % rank)
                break
        comm.send(None, dest=0, tag=tags.EXIT)


if __name__ == '__main__':
    # Choose one of the following:
    # sim_name = 'stick_generic'  # Figure 8CD
    # sim_name = 'hay_generic'    # Figure 4, 6, 7
    # sim_name = 'hay_classic'    # Figure 1, 2, 3
    # sim_name = 'hbp_classic'    # Figure 8AB
    sim_name = 'generic_multimorph'     # Control simulation

    if sim_name == "stick_generic":
        from param_dicts import stick_population_params as param_dict
    elif sim_name == "hay_generic":
        from param_dicts import generic_population_params as param_dict
    elif sim_name == "hay_classic":
        from param_dicts import classic_population_params as param_dict
    elif sim_name == "hbp_classic":
        from param_dicts import hbp_population_params as param_dict
    elif sim_name == "generic_multimorph":
        from param_dicts import multimorph_population_params as param_dict
    else:
        raise RuntimeError("Unrecognized conductance")

    if len(sys.argv) >= 3:
        cell_number = int(sys.argv[3])
        input_region = sys.argv[1]
        correlation = float(sys.argv[2])
        if 'generic' in sim_name:
            distribution = sys.argv[4]
            if sys.argv[5] == "None":
                mu = None
            else:
                mu = float(sys.argv[5])
            param_dict.update({'input_region': input_region,
                               'cell_number': cell_number,
                               'mu': mu,
                               'distribution': distribution,
                               'correlation': correlation,
                               })
        else:
            conductance_type = sys.argv[4]
            holding_potential = None if sys.argv[5] == 'None' else int(sys.argv[5])
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
        if 'generic' in sim_name:
            PopulationMPIgeneric(param_dict)
        else:
            PopulationMPIclassic(param_dict)
