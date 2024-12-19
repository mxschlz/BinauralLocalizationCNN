import sys
from math import ceil
from pathlib import Path
from random import random
from time import sleep

import numpy as np
import pytest
import slab
import matplotlib.pyplot as plt

# raw_stim = slab.Sound(Path('resources/uso_500ms_raw/uso_500ms_2.wav'))
# raw_stim.play()

# for i in range(-180, 180, 10):
#     spatialized_stim = slab.Room(source=slab.HRTF._get_coordinates([i, 0, 1.4], 'interaural').vertical_polar).hrir().resample(48000).apply(raw_stim)
#     spatialized_stim.play()
# -> Makes it start behind (I think to perceive) and goes to right side first from listener's perspective
# But it should start from behind and go left first, no?
hrtf = slab.HRTF('resources/hrtfs/hrtf_b_nh2.sofa')
for i in range(0, 360, 10):
    raw_stim = slab.Sound(Path(f'resources/uso_500ms_raw/uso_500ms_{ceil(random()*30)}.wav'))
    spatialized_stim = slab.Room(source=[i, 0, 1.4]).hrir(hrtf=hrtf).resample(48000).apply(raw_stim)
    spatialized_stim.play(blocking=False)
    sleep(0.3)
sys.exit()

