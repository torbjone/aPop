from __future__ import division

import sys

def PopulationMPIgeneric():
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
    from param_dicts import generic_population_params as param_dict
    if size == 1:
        print "Can't do MPI with one core!"
        sys.exit()

    if rank == 0:

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)
        task = 0
        num_tasks = 100

        for task_idx in range(0, num_tasks):
            task += 1
            sent = False
            while not sent:
                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                source = status.Get_source()
                tag = status.Get_tag()
                if tag == tags.READY:
                    comm.send([task_idx], dest=source, tag=tags.START)
                    print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                    sent = True
                elif tag == tags.DONE:
                    print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                    # pass
                elif tag == tags.ERROR:
                    print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                    for worker in range(1, num_workers + 1):
                        comm.send([None], dest=worker, tag=tags.EXIT)
                    sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        import time
        while True:

            comm.send(None, dest=0, tag=tags.READY)
            [param] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work on task %d\033[0m" % (rank, param)
                time.sleep(1)
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)

if __name__ == '__main__':

    PopulationMPIgeneric()
