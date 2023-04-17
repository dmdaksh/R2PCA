# %%

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

# %%

def load_data(addr, skip_frames=0, scale=1):
    # check number of frames
    num_frames = len(os.listdir(addr))

    # check each frame size
    frame_shape = cv2.imread(addr + "frames_0.png", cv2.IMREAD_GRAYSCALE).shape
    # take top right corner
    frame_shape = (frame_shape[0]//2, frame_shape[1]//2)

    # create data matrix
    data = np.zeros(((frame_shape[0]//scale) * (frame_shape[1]//scale), num_frames //(skip_frames + 1) + 1))

    # load data
    for i in range(0, num_frames, skip_frames + 1):
        frame = cv2.imread(addr + "frames_{}.png".format(i), cv2.IMREAD_GRAYSCALE)
        # take top right corner
        frame = frame[:frame_shape[0], frame_shape[1]:]
        # interpolate
        frame = cv2.resize(frame, (frame_shape[1] // scale, frame_shape[0] // scale))
        data[:, i // (skip_frames + 1)] = frame.reshape(-1)

    # size of data in MB
    print(f"Data size: {data.nbytes / 1024 / 1024} MB")
    return data/255.0, frame_shape

# load data
addr = "./pond-water/"
scale = 2
skip_frames = 3
data, frame_shape = load_data(addr, skip_frames=skip_frames, scale=scale)
print(f'data shape: {data.shape}')
print(f'rank of data: {np.linalg.matrix_rank(data)}')


# %%

# load coeffs.npy and U.npy
U = np.load("U.npy")
coeffs = np.load("coeffs.npy")

# %%

# reconstruct data
L = U @ coeffs
L = L.T
# sparse component
S = data - L
S = S

# %%

# plt.figure(figsize=(20, 10))
for i in range(0, data.shape[1]):
    plt.clf()
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("M")
    plt.imshow(data[:, i].reshape(frame_shape[0] // scale, frame_shape[1] // scale), cmap="gray")
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("L")
    plt.imshow(L[:, i].reshape(frame_shape[0] // scale, frame_shape[1] // scale), cmap="gray")
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("S")
    plt.imshow(S[:, i].reshape(frame_shape[0] // scale, frame_shape[1] // scale), cmap="gray")
    plt.pause(0.05)
