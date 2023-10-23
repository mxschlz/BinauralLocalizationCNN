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
    import slab
    from stim_util import zero_padding
    sig_bi = slab.Binaural.vowel(duration=1.0, samplerate=48000)
    sig_bi = zero_padding(sig_bi, type="front", goal_duration=2.0)
    sig_norm = normalize_binaural_stim(sig_bi.data, sig_bi.samplerate)
    subbands = cochleagram_wrapper(sig_norm[0], sig_norm[1], sliced=True, minimum_padding=0.45)
