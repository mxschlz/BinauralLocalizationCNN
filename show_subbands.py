from CNN_preproc import normalize_binaural_stim, cochleagram_wrapper
import numpy as np


def show_subbands(sig_bi):
    import matplotlib.pyplot as plt

    sig_norm = normalize_binaural_stim(sig_bi.data,
                                       sig_bi.samplerate)
    subbands = cochleagram_wrapper(sig_norm[0], sig_norm[1], sliced=True)
    fig, ax = plt.subplots(3)
    t = np.arange(sig_bi.n_samples) / sig_bi.samplerate
    ax[0].plot(sig_norm[0][0])
    ax[0].plot(sig_norm[0][1])
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
    sns.set_theme(style="white")
    plt.rcParams['text.usetex'] = True  # TeX rendering

    pos_azi = 90
    pos_ele = 30
    samplerate = 48828
    sig = slab.Sound.dynamic_tone(samplerate=samplerate)
    sig = zero_padding(sig, type="frontback", goal_duration=2.1)
    sig_bi = render_stims(orig_stim=sig, pos_elev=pos_ele, pos_azim=pos_azi)
    sound_bi = slab.Binaural(data=sig_bi[0]["sig"], samplerate=samplerate)

    sig_norm = normalize_binaural_stim(sound_bi.data,
                                       sound_bi.samplerate)
    subbands = cochleagram_wrapper(sig_norm[0], sig_norm[1], sliced=True)

    mosaic = """
    ad
    bb
    cc"""
    fig, ax = plt.subplot_mosaic(mosaic=mosaic)
    ax["a"].plot(sig_norm[0][0])
    ax["a"].plot(sig_norm[0][1])
    ax["a"].set_ylabel('Amplitude (arb. unit)')
    ax["a"].set_title("Original signal")
    ax["a"].set_xticklabels([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25])
    ax["a"].legend(['left', 'right'])

    ax["b"].imshow(subbands[:, :, 0], vmin=0, vmax=0.002, aspect='auto')
    ax["b"].set_ylabel('Subbands')
    ax["b"].set_title('Left ear')
    ax["b"].tick_params(axis='x', labelbottom=False)

    ax["c"].imshow(subbands[:, :, 1], vmin=0, vmax=0.002, aspect='auto')
    ax["c"].set_ylabel('Subbands')
    ax["c"].set_title('Right ear')
    ax["c"].set_xticklabels([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25])
    ax["c"].set_xlabel('time (s)')

    sound_bi.spectrum(axis=ax["d"])
    ax["a"].get_shared_x_axes().join(ax["b"], ax["c"])
    plt.show()
