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
# from scipy.signal import coherence
import scipy.fftpack as sf


def average_coherence(signals, timestep, smooth):
    # print "Calculating average coherence (Leski et al., 2013)"
    N_cells = signals.shape[0]
    sample_freq = sf.fftfreq(signals.shape[-1], d=timestep / 1000.)
    pidxs = np.where(sample_freq >= 0)
    # xfft = sf.fft(signals, axis=1)
    # TODO: MAKE THIS PER ELECTRODE INSTEAD???
    xfft = sf.fft(signals, axis=2)
    # print "FFT Done", xfft.shape
    xfft_norm = xfft / (np.absolute(xfft))
    freqs = sample_freq[pidxs]
    xfft_norm_sum = xfft_norm.sum(axis=0)
    c_phi = (np.absolute(xfft_norm_sum) ** 2 - N_cells) / (N_cells * (N_cells - 1))

    if smooth is None:
        return freqs, c_phi[:, pidxs]
    else:
        smooth_freq = np.convolve(np.ones(smooth, 'd')/smooth, freqs, mode='valid')
        smooth_c_phi = np.zeros((signals.shape[1], len(smooth_freq)))
        for elec in range(signals.shape[1]):
            smooth_c_phi[elec, :] = np.convolve(np.ones(smooth, 'd')/smooth, c_phi[elec, pidxs[0]], mode='valid')
        return smooth_freq, smooth_c_phi


def plot_sig_to_elec_grid(x, y):

    top_idxs = np.arange(0, 30, 5)
    middle_idxs = np.arange(30, 60, 5)
    bottom_idxs = np.arange(60, 90, 5)

    for col in range(len(top_idxs)):
        plt.subplot(3, len(top_idxs), col + 1, ylim=[1e-4, 1e1])
        plt.loglog(x, y[top_idxs[col]])
        # plt.loglog(x2, y2[top_idxs[col]])

        plt.subplot(3, len(top_idxs), len(top_idxs) + col + 1, ylim=[1e-4, 1e1])
        plt.loglog(x, y[middle_idxs[col]])
        # plt.loglog(x2, y2[middle_idxs[col]])

        plt.subplot(3, len(top_idxs), len(top_idxs) * 2 + col + 1, ylim=[1e-4, 1e1])
        plt.loglog(x, y[bottom_idxs[col]])
        # plt.loglog(x2, y2[bottom_idxs[col]])


def load_all_signals(num_cells, param_dict):
    num_tsteps = round(param_dict['end_t']/param_dict['timeres_python'] + 1)
    s = np.zeros((num_cells, len(param_dict['electrode_parameters']['x']), num_tsteps))
    for cell_idx in range(num_cells):
        param_dict['cell_number'] = cell_idx
        ns = NeuralSimulation(**param_dict)
        s[cell_idx, :, :] = np.load(join(ns.sim_folder, 'sig_%s.npy' % ns.sim_name))
    return s


def calculate_average_coherence(param_dict, mu, input_region, distribution, correlation):
    param_dict['mu'] = mu
    param_dict['input_region'] = input_region
    param_dict['distribution'] = distribution
    param_dict['correlation'] = correlation
    param_dict['cell_number'] = 0

    ns = NeuralSimulation(**param_dict)
    name = ns.population_sim_name

    if os.path.isfile(join(ns.sim_folder, 'trick_coherence_%s.npy' % name)):
        print "Skipping ", name
        return

    # num_cells = 1000
    num_cells = 1000
    # num_pairs = 1000
    # num_pairs = 100
    print name
    # normalize = lambda sig: (sig - np.average(sig, axis=1).reshape(90, 1))/np.std(sig, axis=1).reshape(90, 1)

    signals = load_all_signals(num_cells, param_dict)
    freq_c_phi, c_phi = average_coherence(signals, ns.timeres_python, 15)

    # pairs = np.random.random_integers(num_cells, size=(num_pairs, 2)) - 1
    # c_mean = np.array([])
    # freq = []
    # counter = 0
    # for idx, pair in enumerate(pairs):
    #     # print idx, pair
    #     if pair[0] == pair[1]:
    #         # print "EQUAL signals"
    #         continue
    #     counter += 1
    #     param_dict['cell_number'] = pair[0]
    #     ns1 = NeuralSimulation(**param_dict)
    #     s1 = np.load(join(ns1.sim_folder, 'sig_%s.npy' % ns1.sim_name))
    #     s1 = normalize(s1)
    #
    #     param_dict['cell_number'] = pair[1]
    #     ns2 = NeuralSimulation(**param_dict)
    #     s2 = np.load(join(ns1.sim_folder, 'sig_%s.npy' % ns2.sim_name))
    #     s2 = normalize(s2)
    #     c_per_elec = []
    #     for elec in range(s1.shape[0]):
    #         freq, c = coherence(s1[elec, :], s2[elec, :], **ns1.welch_dict_scipy)
    #         c_per_elec.append(c)
    #
    #     c_per_elec = np.array(c_per_elec)
    #     if idx == 0:
    #         c_mean = c_per_elec
    #     else:
    #         c_mean[:, :] += c_per_elec
    # c_mean /= counter

    plt.close('all')
    fig = plt.figure()
    plot_sig_to_elec_grid(freq_c_phi, c_phi)
    plt.savefig(join(ns.figure_folder, 'trick_coherence_%s.png' % name))
    np.save(join(ns.sim_folder, 'trick_coherence_%s.npy' % name), c_phi)
    np.save(join(ns.sim_folder, 'trick_coherence_freq_%s.npy' % name), freq_c_phi)

secs = ['homogeneous', 'distal_tuft', 'basal']
mus = [-0.5, 0.0, 2.0]
dists = ['uniform', 'linear_increase', 'linear_decrease']
correlations = [0., 0.1, 1.0]
from param_dicts import generic_population_params
for input_region in secs:
    for distribution in dists:
        for mu in mus:
            for correlation in correlations:
                calculate_average_coherence(generic_population_params, mu, input_region, distribution, correlation)
