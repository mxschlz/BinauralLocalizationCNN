import math
import numpy as np


# this script calculates all the possible combinations of audio samples in the MSL experiment. In total, we have
# 8 unique talkers with 13 audio samples each == 104 total audio samples. Furthermore, we have 7 different speakers
# and between 2 and 6 presented signals in a trial.

# 1. Determine the total number of ways to choose the specified number of talkers from the pool of 8 talkers.
# This can be done using the following formula: nCr = n! / r!(n - r)! --> n = talkers total, r = talkers sample
# 2. Determine the total number of ways to assign each talker to one of the 7 speakers in the room.
# This can be done using the following formula: 7 ^ r --> r = n talkers
# 3. Multiply the two values from steps 1 and 2 to get the total number of combinations for the given condition.

n_talkers = 8
n_countries = 13
n_speakers = 7
conditions = [2, 3, 4, 5, 6]
n_reps = 5  # TODO: find proper value for this
combinations = list()  # save results


def ncr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


for cond in conditions:
    comb = ncr(n_talkers, cond)
    combinations.append(comb)
    print(f"Possible combinations for {cond} talkers: ", comb)

print(f"Total combinations: {np.sum(combinations)}")
print(f"Total combinations repeated {n_reps} times: {np.sum(combinations) * n_reps}")
