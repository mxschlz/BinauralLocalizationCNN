import numpy as np


def decide_sound_presence(cond_dis, criterion=0.09):
    n_sounds_present = list()
    for cd in cond_dis:
        max_prob = cond_dis.max()
        rule = max_prob * criterion
        true_idx = np.where(cd >= rule)
        n_sounds_present.append(len(true_idx[0]))
    return n_sounds_present