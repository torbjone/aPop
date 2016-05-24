import pylab as plt
import numpy as np
import scipy.fftpack as ff
from matplotlib import mlab as ml

dt = 2**-3
end_t = 2**10 - dt
num_tsteps = round(end_t/dt + 1)

print num_tsteps
x = np.arange(num_tsteps) * dt
sample_freq = ff.fftfreq(x.shape[0], d=dt/1000.)
pidxs = np.where(sample_freq >= 0)
freqs = sample_freq[pidxs]

input_amp = 1. / ((freqs - 50) ** 2 + 2) * 1. / (freqs + .05)**2

y = np.sum([input_amp[idx] * np.sin(2 * np.pi * freqs[idx] * x / 1000. + np.random.random() * 2 * np.pi)
            for idx in range(len(input_amp))], axis=0)

y += np.random.normal(0, 0.0001, size=len(y))

divide_into_welch = 4. # 8.
welch_dict = {'Fs': 1000. / dt ,
                   'NFFT': int(num_tsteps/divide_into_welch),
                   'noverlap': int(num_tsteps/divide_into_welch/2.),
                   'window': plt.window_hanning,
                   'detrend': plt.detrend_mean,
                   'scale_by_freq': False,
                   }

mlab_psd, mlab_freqs = ml.psd(y, **welch_dict)

print len(freqs), len(mlab_freqs)
print freqs
print mlab_freqs
Y = ff.fft(y)
ff_psd = np.abs(Y[pidxs])**2 / len(Y[pidxs])


def smooth_signal(new_x, old_x, y):
    new_y = np.zeros(len(new_x))
    df = new_x[1] - new_x[0]
    for m, mfreq in enumerate(new_x):
        new_y[m] = np.average(y[(old_x >= mfreq - df/2) & (old_x < mfreq + df/2)])
    return new_y

bin_psd = smooth_signal(mlab_freqs, freqs, ff_psd)

smooth = 5
smooth_freq = np.convolve(np.ones(smooth, 'd')/smooth, freqs, mode='valid')
smooth_psd = np.convolve(np.ones(smooth, 'd')/smooth, ff_psd, mode='valid')


plt.subplot(211)
plt.loglog(freqs, input_amp**2 * len(input_amp), 'k')
plt.loglog(freqs, ff_psd, 'b')
plt.loglog(mlab_freqs, mlab_psd * len(mlab_psd), 'r')
plt.loglog(mlab_freqs, bin_psd, 'g')
plt.loglog(smooth_freq, smooth_psd, 'pink')
plt.subplot(212)
plt.plot(x, y)

plt.show()
