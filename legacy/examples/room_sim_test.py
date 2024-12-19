import os

import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

data_path = os.path.join('../..', 'tfrecords', 'origin')
wav_ori = 'test.wav'
wf = os.path.join(data_path, wav_ori)
sf, sig_ori = wavfile.read(wf)
# just need a single channel for the room simulation
sig_ori = sig_ori[:, 0]

# generate a box room
# corners = np.array([[0, 0], [0, 3], [5, 3], [5, 1], [3, 1], [3, 0]]).T  # [x,y]
# room = pra.Room.from_corners(corners, fs=sf, ray_tracing=True, air_absorption=True)
rm_size = [5, 4, 3.2]
room = pra.ShoeBox(rm_size, fs=sf, max_order=4, materials=pra.Material(0.3, 0.15),
                   ray_tracing=True, air_absorption=True)
# Set the ray tracing parameters
room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)
# add source and set the signal to WAV file content
room.add_source([1., 2., 1.2], signal=sig_ori)
# generate 2 mic for the ears
# add two-microphone array
ears = np.array([[3.5, 3.5], [1.88, 2.12], [1.2, 1.2]])  # [[x], [y], [z]]
room.add_microphone(ears)

# compute image sources
room.image_source_model()

# visualize 3D polyhedron room and image sources
fig, ax = room.plot(img_order=3)
fig.set_size_inches(18.5, 10.5)

fig, ax = room.plot()
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 6])
