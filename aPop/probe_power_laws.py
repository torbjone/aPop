from plotting_convention import *
import pylab as plt
import numpy as np
import os
import sys
from os.path import join
from suppress_print import suppress_stdout_stderr
with suppress_stdout_stderr():
    import LFPy
from NeuralSimulation import NeuralSimulation
import tools
import scipy.fftpack as sf
import matplotlib
import scipy.fftpack as ff
from matplotlib import mlab as ml


def return_freq_and_psd(tvec, sig):
    """ Returns the power and freqency of the input signal"""

    timestep = tvec[1] - tvec[0]
    sample_freq = ff.fftfreq(len(sig), d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    Y = ff.fft(sig)[pidxs]
    power = np.abs(Y /len(Y))**2

    return freqs, power




def probe_power_laws():

    f = 16.

    t = np.arange(2**14) * 2**-14

    f_array = np.linspace(f, f, len(t))
    f_array[len(t)/5:] = f + 0.1
    sig_dict = {}
    psd_dict = {}

    sig_dict['sine'] = np.sin(2 * np.pi * f * t)
    sig_dict['sine_shift'] = np.sin(2 * np.pi * f * t) + 1.5 * (t - t[len(t)/2])
    sig_dict['sine_pause'] = np.zeros(len(t))
    sig_dict['sine_pause'][100:] = np.sin(2 * np.pi * f * t[:-100])
    sig_dict['freq_mod'] = np.sin(2 * np.pi * f_array * t)
    # sig_dict['sine2'] = np.sin(2 * np.pi * f * t)**2
    sig_dict['square'] = np.ones(len(t))
    sig_dict['square'][sig_dict['sine'] < 0] = -1
    # sig_dict['shift'] = -np.ones(len(t))*0.2
    # sig_dict['shift'][t > t[len(t)/2]] = 0.2
    # sig_dict['linear'] = 0.7 * (t - np.mean(t))

    noise = np.random.normal(0, .25, size=len(t))

    for key, sig in sig_dict.items():
        sig += noise

    for key, sig in sig_dict.items():
        freqs, psd_dict[key] = return_freq_and_psd(t, sig)
        # psd_dict[key], freqs = ml.psd(sig, Fs=len(t) / t[-1], NFFT=len(t)/1)

    fig = plt.figure(figsize=[18, 9])

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)#, ylim=[1e-3, 1e1])

    lines = []
    line_names = []
    for key, sig in sig_dict.items():
        l, = ax1.plot(t, sig, lw=2)
        l, = ax2.loglog(freqs, psd_dict[key], '-x', lw=2)
        lines.append(l)
        line_names.append(key)
        # print np.sum(psd_dict[key])

    simplify_axes(fig.axes)
    fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
    fig.savefig('prope_powerlaws_no_welch_with_noise.png')
    plt.show()

if __name__ == '__main__':

    probe_power_laws()
