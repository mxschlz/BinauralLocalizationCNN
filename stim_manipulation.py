import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import slab
import pickle
import numpy as np


_ils = pickle.load(open('interaural_level_spectrum/ils.pickle', 'rb'))  # load interaural level spectrum


def change_itd(stim, azi):
    if not isinstance(stim, slab.Binaural):
        raise ValueError("Stimulus must be instance of slab.Binaural")
    itd = slab.Binaural.azimuth_to_itd(azi)
    itd_ns = int(round(itd * stim.samplerate))
    return stim.itd(itd_ns)


def change_ild(stim, azi):
    if not isinstance(stim, slab.Binaural):
        raise ValueError("Stimulus must be instance of slab.Binaural")
    ild = slab.Binaural.azimuth_to_ild(azi, ils=_ils)
    return stim.ild(stim.ild() + ild)


if __name__ == "__main__":
    stim = slab.Binaural.whitenoise()
    plt.plot(stim.data[:, 0])
    plt.plot(stim.data[:, 1])
    stim.itd()
    stim.ild()
    azi = 45  # azimuth in degrees
    stim = change_itd(stim, azi)
    stim = change_ild(stim, azi)
