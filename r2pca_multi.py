import os
import time

import cv2
import numpy as np
import scipy.linalg as la


def load_data(addr, skip_frames=0, scale=1):
    # check number of frames
    num_frames = len(os.listdir(addr))

    # check each frame size
    frame_shape = cv2.imread(addr + "frames_0.png", cv2.IMREAD_GRAYSCALE).shape
    # take top right corner
    frame_shape = (frame_shape[0] // 2, frame_shape[1] // 2)

    # create data matrix
    data = np.zeros(
        (
            (frame_shape[0] // scale) * (frame_shape[1] // scale),
            num_frames // (skip_frames + 1) + 1,
        )
    )

    # load data
    for i in range(0, num_frames, skip_frames + 1):
        frame = cv2.imread(addr + "frames_{}.png".format(i), cv2.IMREAD_GRAYSCALE)
        # take top right corner
        frame = frame[: frame_shape[0], frame_shape[1] :]
        # interpolate
        frame = cv2.resize(frame, (frame_shape[1] // scale, frame_shape[0] // scale))
        data[:, i // (skip_frames + 1)] = frame.reshape(-1)

    # size of data in MB
    print(f"Data size: {data.nbytes / 1024 / 1024} MB")
    return data, frame_shape

def compute_U(self):

    A = np.zeros((self.d, self.d - self.rank))

    for i in range(self.d - self.rank):
        # selecting random (r+1)x(r+1) entries of M
        oi = np.random.randint(0, self.d, self.rank + 1)
        oj = np.random.randint(0, self.N, self.rank + 1)

        if np.linalg.matrix_rank(self.data[oi, :][:, oj]) == self.rank:
            # get null space of M[oi, :][:, oj]
            aoi = la.null_space(self.data[oi, :][:, oj].T).reshape(-1)
            A[oi, i] = aoi

    U_hat = la.null_space(A.T)
    return U_hat



def compute_theta(M, U):
    d, N = M.shape
    r = U.shape[1]

    # coeffs matrix
    coeffs = np.zeros((r, N))

    for itr in range(N):
        print(f"iteration: {itr} / {N}")
        # flag to check if we already found r+1 uncorrupted entries of column i
        flag = False

        while not flag:

            # selecting random (r+1) entries of column i
            oi = np.random.randint(0, d, r + 1)

            Uoi = U[oi, :]
            Moi = M[oi, itr]
            coeffs[:, itr] = np.linalg.lstsq(Uoi, Moi, rcond=None)[0]

            # check objective
            obj = np.linalg.norm(Moi - Uoi @ coeffs[:, itr])

            termination = np.linalg.norm(obj) / np.linalg.norm(Moi) < 1e-3

            if termination:
                flag = True

    return coeffs


if __name__ == "__main__":

    # load data
    addr = "./pond-water/"
    scale = 10
    skip_frames = 10
    data, frame_shape = load_data(addr, skip_frames=skip_frames, scale=scale)
    print(f"data shape: {data.shape}")
    print(f"rank of data: {np.linalg.matrix_rank(data)}")

    rank = 5
    r2pca = R2PCA(data.T, rank)
    U, coeffs = r2pca.run()
    # U, coeffs = run(data.T, rank)

    # save U and coeffs
    np.save("U.npy", U)
    np.save("coeffs.npy", coeffs)
