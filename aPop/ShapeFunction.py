from __future__ import division
import param_dicts
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False
import os
from os.path import join
import sys
import numpy as np
from NeuralSimulation import NeuralSimulation
import tools

def sum_shape_functions_classic(param_dict):

    secs = ['homogeneous', 'distal_tuft', 'basal']
    conductance_types = ['active', 'passive', 'Ih_frozen']
    holding_potentials = [-80, -60]

    num_cells = 200
    freqs = np.zeros(10001)
    for input_region in secs:
        for conductance_type in conductance_types:
            for holding_potential in holding_potentials:
                average_psd = np.zeros((len(param_dict['electrode_parameters']['x']), 2**10))
                cell_count = 0
                for cell_number in xrange(num_cells):
                    param_dict.update({'input_region': input_region,
                                                             'cell_number': cell_number,
                                                             'conductance_type': conductance_type,
                                                             'holding_potential': holding_potential})
                    ns = NeuralSimulation(**param_dict)
                    lfp = 1000 * np.load(join(ns.sim_folder, 'sig_%s.npy' % ns.sim_name))
                    freqs, lfp_psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[:, :])
                    if not average_psd.shape == lfp_psd.shape:
                        print "Reshaping", lfp_psd.shape, freqs.shape, freqs
                        average_psd = np.zeros(lfp_psd.shape)
                    # if not welch_freqs.shape == freqs.shape:
                    #     print "Reshaping frequency"
                    # welch_freqs = freqs
                    average_psd += lfp_psd
                    cell_count += 1
                F = average_psd / cell_count

                ns.plot_F(freqs, F)

                np.save(join(ns.sim_folder, 'F2_%s.npy' % ns.population_sim_name), F)
    np.save(join(ns.sim_folder, 'freqs.npy'), freqs)


def sum_shape_function(param_dict, input_region, distribution, mu, num_cells):
    average_psd = np.zeros((len(param_dict['electrode_parameters']['x']), 2**10))
    cell_count = 0
    freqs = []
    for cell_number in xrange(num_cells):
        param_dict.update({'input_region': input_region,
                           'cell_number': cell_number,
                           'mu': mu,
                           'distribution': distribution})
        ns = NeuralSimulation(**param_dict)
        lfp = 1000 * np.load(join(ns.sim_folder, 'sig_%s.npy' % ns.sim_name))
        freqs, lfp_psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[:, :])
        if not average_psd.shape == lfp_psd.shape:
            # print "Reshaping", lfp_psd.shape, freqs.shape, freqs
            average_psd = np.zeros(lfp_psd.shape)

        average_psd += lfp_psd
        cell_count += 1
    F2 = average_psd / cell_count
    ns = NeuralSimulation(**param_dict)
    np.save(join(ns.sim_folder, 'F2_%s.npy' % ns.population_sim_name), F2)
    ns.plot_F(freqs, F2)
    np.save(join(ns.sim_folder, 'freqs.npy'), freqs)

def sum_all_shape_functions_generic(param_dict):

    num_cells = 200 if at_stallo else 5
    for input_region in param_dict['input_regions']:
        for distribution in param_dict['distributions']:
            for mu in param_dict['mus']:
                sum_shape_function(param_dict, input_region, distribution, mu, num_cells)


def ShapeFunctionMPIgeneric(param_dict):
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
        num_cells = 100 if at_stallo else 5
        num_tasks = len(param_dict['input_regions']) * len(param_dict['mus']) * len(param_dict['distributions']) * (num_cells - 0)
        for input_region in param_dict['input_regions']:
            for distribution in param_dict['distributions']:
                for mu in param_dict['mus']:
                    for cell_idx in xrange(0, num_cells):
                        task += 1
                        sent = False
                        while not sent:
                            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                            source = status.Get_source()
                            tag = status.Get_tag()
                            if tag == tags.READY:
                                comm.send([input_region, distribution, mu, cell_idx], dest=source, tag=tags.START)
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


def ShapeFunctionMPIclassic(param_dict):
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
        conductance_types = ['active', 'passive', 'Ih_frozen']
        holding_potentials = [-80, -60]

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)
        task = 0
        num_cells = 10
        num_tasks = len(secs) * len(holding_potentials) * len(conductance_types) * (num_cells)

        for input_sec in secs:
            for conductance_type in conductance_types:
                for holding_potential in holding_potentials:
                    for cell_idx in xrange(0, num_cells):
                        task += 1
                        sent = False
                        while not sent:
                            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                            source = status.Get_source()
                            tag = status.Get_tag()
                            if tag == tags.READY:
                                comm.send([input_sec, conductance_type, holding_potential, cell_idx], dest=source, tag=tags.START)
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
            [input_sec, conductance_type, holding_potential, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work on %s %s %+1.1f cell %d\033[0m" % (rank, input_sec, conductance_type,
                                                                                 holding_potential, cell_idx)
                try:
                    #print "python %s %1.1f %s %s %d" % (sys.argv[0], mu, input_sec, channel_dist, cell_idx)
                    os.system("python %s %s %s %d %d" % (sys.argv[0], conductance_type, input_sec, holding_potential, cell_idx))
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

    if len(sys.argv) == 5:
        cell_number = int(sys.argv[4])
        input_region = sys.argv[2]
        if conductance == 'generic':
            from param_dicts import shape_function_params
            mu = float(sys.argv[1])
            distribution = sys.argv[3]
            shape_function_params.update({'mu': mu, 'input_region': input_region,
                      'distribution': distribution, 'cell_number': cell_number}) #mu, input_sec, channel_dist, cell_idx
            ns = NeuralSimulation(**shape_function_params)
        else:
            from param_dicts import distributed_delta_classic_params
            conductance_type = sys.argv[1]
            holding_potential = int(sys.argv[3])
            distributed_delta_classic_params.update({'input_region': input_region,
                                                     'cell_number': cell_number,
                                                     'conductance_type': conductance_type,
                                                     'holding_potential': holding_potential})
            ns = NeuralSimulation(**distributed_delta_classic_params)
        ns.run_distributed_synaptic_simulation()
    elif len(sys.argv) == 2 and sys.argv[1] == 'MPIgeneric':
        from param_dicts import shape_function_params
        ShapeFunctionMPIgeneric(shape_function_params)
    elif len(sys.argv) == 2 and sys.argv[1] == 'MPIclassic':
        from param_dicts import distributed_delta_classic_params
        ShapeFunctionMPIclassic(distributed_delta_classic_params)
    elif len(sys.argv) == 2 and sys.argv[1] == 'sum':
        if conductance == 'generic':
            from param_dicts import shape_function_params
            sum_all_shape_functions_generic(shape_function_params)
        else:
            from param_dicts import distributed_delta_classic_params
            sum_shape_functions_classic(distributed_delta_classic_params)
