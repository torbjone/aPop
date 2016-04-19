import os
from os.path import join
import numpy as np
import pylab as plt
from matplotlib import mlab
from scipy import signal
import scipy.fftpack as sf


def average_coherence(signals, timestep, smooth):

    N_cells = signals.shape[0]
    sample_freq = sf.fftfreq(signals.shape[1], d=timestep / 1000.)
    pidxs = np.where(sample_freq >= 0)
    xfft = sf.fft(signals, axis=1)
    xfft_norm = xfft / (np.absolute(xfft))
    freqs = sample_freq[pidxs]
    xfft_norm_sum = xfft_norm.sum(axis=0)
    phibar = (np.absolute(xfft_norm_sum) ** 2 - N_cells) / (N_cells * (N_cells - 1))
    if smooth is None:
        return freqs, phibar[pidxs]
    else:
        return np.convolve(np.ones(smooth, 'd')/smooth, freqs, mode='valid'), np.convolve(np.ones(smooth, 'd')/smooth, phibar[pidxs], mode='valid')


def make_signal(num_cells, num_tsteps, correlation):
    signal = np.zeros((num_cells, num_tsteps))
    common_signal = np.random.normal(0, 1, size=num_tsteps)
    for cell_idx in range(num_cells):
        cell_signal = np.random.normal(0, 1, size=num_tsteps)
        signal[cell_idx, :] = (1 - np.sqrt(correlation)) * cell_signal + np.sqrt(correlation) * common_signal
    return signal


def return_mean_coherence(s, num_cells, num_pairs):

    pairs = np.random.random_integers(num_cells, size=(num_pairs, 2)) - 1
    c_mean_mlab = np.array([])
    c_mean_scipy = np.array([])
    freq_mlab = []
    freq_scipy = []
    counter = 0
    for idx, pair in enumerate(pairs):
        if pair[0] == pair[1]:
            continue
        counter += 1
        freq_scipy, c_scipy = signal.coherence(s[pair[0], :], s[pair[1], :], **welch_scipy)
        c_mlab, freq_mlab = mlab.cohere(s[pair[0], :], s[pair[1], :], **welch_mlab)
        if idx == 0:
            c_mean_mlab = c_mlab
            c_mean_scipy = c_scipy
        else:
            c_mean_mlab[:] += c_mlab
            c_mean_scipy[:] += c_scipy

    c_mean_mlab /= counter
    c_mean_scipy /= counter
    return freq_mlab, c_mean_mlab, freq_scipy, c_mean_scipy

num_tsteps = 10000
num_cells = 1000
num_pairs = 1000
correlation = .01

s = make_signal(num_cells, num_tsteps, correlation)
normalize = lambda sig: (sig - np.average(sig[:, :, None], axis=1))/np.std(sig[:, :, None], axis=1)

s = normalize(s)

timeres_python = 2**-3

divide_into_welch = 10. # 8.

welch_mlab = {'Fs': 1000 / timeres_python,
               'NFFT': int(num_tsteps/divide_into_welch),
               'noverlap': int(num_tsteps/divide_into_welch/2.),
               'window': plt.window_hanning,
               'detrend': plt.detrend_mean,
               'scale_by_freq': True,
               }

welch_scipy = {'fs': 1000 / timeres_python,
                   'nfft': int(num_tsteps/divide_into_welch),
                    'window': "hanning",
                   'detrend': 'constant',
                   # 'scale_by_freq': True,
                   }

f_phi, c_phi = average_coherence(s, timeres_python, smooth=20)
print f_phi

freq_mlab, c_mean_mlab, freq_scipy, c_mean_scipy = return_mean_coherence(s, num_cells, num_pairs)


plt.subplot(111, ylim=[1e-5, 2e0])
plt.loglog(f_phi, c_phi, 'g')
plt.loglog(freq_scipy, np.ones(len(freq_scipy)) * correlation, lw=2)
plt.loglog(freq_mlab, (c_mean_mlab), 'k')
plt.loglog(freq_scipy, c_mean_scipy, 'r')
plt.show()