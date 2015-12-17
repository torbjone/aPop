import numpy as np
import scipy.fftpack as ff
from matplotlib import mlab

def return_freq_and_psd(tvec, sig):
    """ Returns the power and freqency of the input signal"""
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:, pidxs[0]]
    power = np.abs(Y)**2/Y.shape[1]
    return freqs, power

def return_freq_and_psd_welch(sig, welch_dict):
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    psd = []
    freqs = None
    for idx in xrange(sig.shape[0]):
        yvec_w, freqs = mlab.psd(sig[idx, :], **welch_dict)
        psd.append(yvec_w)
    return freqs, np.array(psd)
