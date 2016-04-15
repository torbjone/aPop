import os
from os.path import join
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False
import numpy as np
import pylab as plt
from matplotlib import mlab
from NeuralSimulation import NeuralSimulation




def plot_sig_to_elec_grid(x, y, fig):

    top_idxs = np.arange(0, 30, 5)
    middle_idxs = np.arange(30, 60, 5)
    bottom_idxs = np.arange(60, 90, 5)

    for col in range(len(top_idxs)):
        plt.subplot(3, len(top_idxs), col + 1, ylim=[1e-4, 1e1])
        plt.loglog(x, y[top_idxs[col]])

        plt.subplot(3, len(top_idxs), len(top_idxs) + col + 1, ylim=[1e-4, 1e1])
        plt.loglog(x, y[middle_idxs[col]])

        plt.subplot(3, len(top_idxs), len(top_idxs) * 2 + col + 1, ylim=[1e-4, 1e1])
        plt.loglog(x, y[bottom_idxs[col]])

def calculate_average_coherence(param_dict, mu, input_region, distribution, correlation):
    param_dict['mu'] = mu
    param_dict['input_region'] = input_region
    param_dict['distribution'] = distribution
    param_dict['correlation'] = correlation
    param_dict['cell_number'] = 0

    ns = NeuralSimulation(**param_dict)
    num_cells = 1000
    num_pairs = 1000
    name = ns.population_sim_name
    normalize = lambda sig: (sig - np.average(sig, axis=1).reshape(90, 1))/np.std(sig, axis=1).reshape(90, 1)

    pairs = np.random.random_integers(num_cells, size=(num_pairs, 2)) - 1
    c_mean = np.array([])
    plt.close('all')
    counter = 0
    for idx, pair in enumerate(pairs):
        # print idx, pair
        if pair[0] == pair[1]:
            # print "EQUAL signals"
            continue
        counter += 1
        param_dict['cell_number'] = pair[0]
        ns1 = NeuralSimulation(**param_dict)
        s1 = np.load(join(ns1.sim_folder, 'sig_%s.npy' % ns1.sim_name))
        s1 = normalize(s1)

        param_dict['cell_number'] = pair[1]
        ns2 = NeuralSimulation(**param_dict)
        s2 = np.load(join(ns1.sim_folder, 'sig_%s.npy' % ns2.sim_name))
        s2 = normalize(s2)
        c_per_elec = []
        for elec in range(s1.shape[0]):
            c, freq = mlab.cohere(s1[elec, :], s2[elec, :], **ns1.welch_dict)
            c_per_elec.append(c)

        c_per_elec = np.array(c_per_elec)
        # print c_per_elec.shape
        if idx == 0:
            c_mean = c_per_elec
        else:
            c_mean[:, :] += c_per_elec

        # plt.loglog(freq, c_per_elec, lw=0.5, c='gray')

    c_mean /= counter
    # plt.loglog(freq, c_mean, lw=2, c='k')
    # plt.ylim([1e-4, 1e1])
    fig = plt.figure()
    plot_sig_to_elec_grid(freq, c_mean, fig)
    plt.savefig(join(ns.figure_folder, 'coherence_%s.png' % name))
    np.save(join(ns.sim_folder, 'coherence_%s.npy' % name), c_mean)

secs = ['homogeneous', 'distal_tuft', 'basal']
mus = [-0.5, 0.0, 2.0]
dists = ['uniform', 'linear_increase', 'linear_decrease']
from param_dicts import generic_population_params
for input_region in secs:
    for distribution in dists:
        for mu in mus:
            for correlation in [0., 0.1, 1.0]:
                calculate_average_coherence(generic_population_params, mu, input_region, distribution, correlation)
