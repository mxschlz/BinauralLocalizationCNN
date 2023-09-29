from CNN_preproc import normalize_binaural_stim, cochleagram_wrapper
import numpy as np


def show_subbands(sig_bi):
    import matplotlib.pyplot as plt

    sig_norm = normalize_binaural_stim(sig_bi.data,
                                       sig_bi.samplerate)
    subbands = cochleagram_wrapper(sig_norm[0], sig_norm[1], sliced=True)
    fig, ax = plt.subplots(3)
    t = np.arange(48000) / 48000
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
    ax[2].set_xticklabels([-0.21, 0, 0.21, 0.42, 0.63, 0.84])
    ax[2].set_xlabel('time (s)')

    plt.show()
