from CNN_preproc import normalize_binaural_stim, cochleagram_wrapper
import numpy as np


def show_subbands(sig_bi):
    import matplotlib.pyplot as plt

    sig_norm = normalize_binaural_stim(sig_bi.data,
                                       sig_bi.samplerate)
    subbands = cochleagram_wrapper(sig_norm[0], sig_norm[1], sliced=True)
    fig, ax = plt.subplots(3)
    t = np.arange(sig_bi.n_samples) / sig_bi.samplerate
    ax[0].plot(t, sig_norm[0][0][1000:49000])
    ax[0].plot(t, sig_norm[0][1][1000:49000])
    ax[0].set_ylabel('Amplitude (arb. unit)')
    ax[0].set_title("original signal")
    ax[0].legend(['left', 'right'])

    ax[1].imshow(subbands[:, :, 0], vmin=0, vmax=0.002, aspect='auto')
    ax[1].set_ylabel('Subbands')
    ax[1].set_title('left ear')
    ax[1].tick_params(axis='x', labelbottom=False)

    ax[2].imshow(subbands[:, :, 1], vmin=0, vmax=0.002, aspect='auto')
    ax[2].set_ylabel('Subbands')
    ax[2].set_title('right ear')
    # ax[2].set_xticklabels([-0.21, 0, 0.21, 0.42, 0.63, 0.84])
    ax[2].set_xlabel('time (s)')

    plt.show()


if __name__ == "__main__":
    from stim_gen import render_stims
    import slab
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from stim_util import zero_padding
    import seaborn as sns
    import scienceplots

    plt.style.use("science")
    plt.ion()

    pos_azi = 90
    pos_ele = 30
    samplerate = 48000
    sig = slab.Sound.vowel(duration=2.1)
    # sig = zero_padding(sig, type="back", goal_duration=2.1)
    sig_bi = render_stims(orig_stim=sig, pos_elev=pos_ele, pos_azim=pos_azi)
    sound_bi = slab.Binaural(data=sig_bi[0]["sig"], samplerate=samplerate)

    sig_norm = normalize_binaural_stim(sound_bi.data,
                                       sound_bi.samplerate)
    subbands = cochleagram_wrapper(sig_norm[0], sig_norm[1], sliced=True)
    t = np.arange(48000)/48000

    diff = subbands[:, :, 0] - subbands[:, :, 1]

    mosaic = """
    c
    b
    """
    fig, ax = plt.subplot_mosaic(mosaic=mosaic)
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    pos = ax["b"].imshow(diff, vmin=0, vmax=0.0001, aspect='auto', cmap="binary")
    ax["b"].set_ylabel('Subbands', fontsize=10)
    ax["b"].tick_params(axis='x', labelbottom=False)
    ax["b"].set_xticklabels([-0.2, 0, 0.2, 0.4, 0.6, 0.80])
    sound_bi.spectrum(axis=ax["c"])
    ax["c"].set_title("")
    ax["c"].set_xlabel("Frequency [Hz]")
    ax["b"].tick_params(axis='x', labelbottom=True)
    ax["b"].set_xlabel("Time (s)", fontsize=10)
    fig.colorbar(pos, ax=ax["b"])

    plt.savefig("/home/max/labplatform/plots/MA_thesis/materials_methods/waveform_spectrum_cochleagram.png")
