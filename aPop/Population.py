from __future__ import division
import pylab as plt
import numpy as np
import os
from os.path import join
import LFPy
import sys
from NeuralSimulation import NeuralSimulation


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
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 1])

    plt.subplot(222, xlabel='x', ylabel='z', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 2])

    plt.subplot(212, ylabel='Cell number', xlabel='Time', title='Raster plot', frameon=False)
    for idx in range(1000):
        plt.scatter(all_spiketimes[idx], np.ones(len(all_spiketimes[idx])) * idx, edgecolors='none', s=4)

    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'population_%s.png' % param_dict['name']))


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

        secs = ['homogeneous', 'distal_tuft', 'basal']
        mus = [-0.5, 0.0, 2.0]
        dists = ['uniform', 'linear_increase', 'linear_decrease']

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)
        task = 0
        num_cells = 100
        num_tasks = len(secs) * len(mus) * len(dists) * (num_cells)

        for input_sec in secs:
            for channel_dist in dists:
                for mu in mus:
                    for cell_idx in xrange(0, num_cells):
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
                                    comm.send([None, None, None, None], dest=worker, tag=tags.EXIT)
                                sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None, None, None, None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [input_sec, channel_dist, mu, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work on %s %s %+1.1f cell %d\033[0m" % (rank, input_sec,
                                                                                 channel_dist, mu, cell_idx)
                try:
                    #print "python %s %1.1f %s %s %d" % (sys.argv[0], mu, input_sec, channel_dist, cell_idx)
                    os.system("python %s %1.1f %s %s %d" % (sys.argv[0], mu, input_sec, channel_dist, cell_idx))
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
    from param_dicts import small_population_params
    initialize_population(small_population_params)
    plot_population(small_population_params)
    sys.exit()

    # print sys.argv
    if len(sys.argv) == 5:
        from param_dicts import population_params
        ns = NeuralSimulation(**population_params)
        args = [float(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])] #mu, input_sec, channel_dist, cell_idx
        ns.run_distributed_synaptic_simulation(*args)
    elif len(sys.argv) == 2 and sys.argv[1] == 'MPI':
        PopulationMPI()
    elif len(sys.argv) == 2 and sys.argv[1] == 'sum':
        from param_dicts import distributed_delta_params
        ns = NeuralSimulation(**distributed_delta_params)
        ns.sum_population()