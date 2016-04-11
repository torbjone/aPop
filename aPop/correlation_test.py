import os
from os.path import join
import numpy as np
import pylab as plt
from matplotlib import mlab

name = 'sig_generic_population_hay_generic_linear_increase_2.0_distal_tuft_0.00'
name2 = 'sig_generic_population_hay_generic_linear_increase_2.0_distal_tuft_1.00'
folder = join(os.getenv('HOME'), 'work', 'aPop', 'population', 'simulations')
s0 = np.load(join(folder, '%s_00001.npy' % name))[:1, :]
s1 = np.load(join(folder, '%s_00002.npy' % name))[:1, :]
s20 = np.load(join(folder, '%s_00001.npy' % name2))[:1, :]
s21 = np.load(join(folder, '%s_00002.npy' % name2))[:1, :]

s0 = (s0 - np.average(s0)) / np.std(s0)
s1 = (s1 - np.average(s1)) / np.std(s1)
s20 = (s20 - np.average(s20)) / np.std(s20)
s21 = (s21 - np.average(s21)) / np.std(s21)

# a = np.random.normal(0, 0.1, size=(1, 100000))
# b = np.random.normal(0, 0.1, size=(1, 100000))
# c = np.random.normal(0, 0.1, size=(1, 100000))
# corr = .1
# s0 = (1 - corr) * a + corr * c
# s1 = (1 - corr) * b + corr * c
# s0 /= np.std(s0)
# s20 /= np.std(s20)
# s1 /= np.std(s1)
# s21 /= np.std(s21)

# plt.plot(s0[0])
# plt.plot(s1[0])
# plt.show()


timeres_python = 2**-3
num_tsteps = s0.shape[1]
divide_into_welch = 10. # 8.
welch_dict = {'Fs': 1000 / timeres_python,
                   'NFFT': int(num_tsteps/divide_into_welch),
                   'noverlap': int(num_tsteps/divide_into_welch/2.),
                   'window': plt.window_hanning,
                   'detrend': plt.detrend_mean,
                   'scale_by_freq': True,
                   }

print s0.shape

psd0, freqs = mlab.psd(s0[0, :], **welch_dict)
psd20, freqs = mlab.psd(s20[0, :], **welch_dict)
psd1, freqs = mlab.psd(s1[0, :], **welch_dict)
psd21, freqs = mlab.psd(s21[0, :], **welch_dict)

c, f = mlab.cohere(s0[0, :], s1[0, :], **welch_dict)
c2, f2 = mlab.cohere(s20[0, :], s21[0, :], **welch_dict)

# print c
plt.subplot(121)
plt.loglog(freqs, psd0, 'k')
plt.loglog(freqs, psd20, 'g')
plt.loglog(freqs, psd1, 'gray')
plt.loglog(freqs, psd21, 'c')
plt.subplot(122)
plt.loglog(f, np.abs(c), 'k')
plt.loglog(f, np.abs(c2), 'g')
plt.show()