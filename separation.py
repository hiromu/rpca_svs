import librosa
import math
import os
import sys

import numpy as np
from numpy.linalg import norm, svd

# from http://kastnerkyle.github.io/posts/robust-matrix-decomposition/
def inexact_alm_rpca(X, lmbda=.01, tol=1e-7, maxiter=1000, verbose=False):
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])

    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')

    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0

    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)

        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)

        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])

        A = Aupdate
        E = Eupdate
        Z = X - A - E
        Y = Y + mu * Z

        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
            break

    if verbose:
        print("Finished at iteration %d" % (itr))  

    return A, E

def singing_voice_separation(X, fs, lmbda=1, nFFT=1024, gain=1, power=1):
    scf = 2 / 3.0
    S_mix = scf * librosa.core.stft(X, n_fft=nFFT)

    A_mag, E_mag = inexact_alm_rpca(np.power(np.abs(S_mix), power),
                                    lmbda=lmbda / math.sqrt(max(S_mix.shape)))
    PHASE = np.angle(S_mix)

    A = A_mag * np.exp(1j * PHASE)
    E = E_mag * np.exp(1j * PHASE)

    mask = np.abs(E) > (gain * np.abs(A))
    Emask = mask * S_mix
    Amask = S_mix - Emask

    wavoutE = librosa.core.istft(Emask)
    wavoutA = librosa.core.istft(Amask)

    wavoutE /= np.abs(wavoutE).max()
    wavoutA /= np.abs(wavoutA).max()

    return wavoutE, wavoutA

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('%s [input wav] [voice output] [music output]' % sys.argv[0])
        sys.exit(-1)

    data, sr = librosa.load(sys.argv[1])
    E, A = singing_voice_separation(data, sr)
    librosa.output.write_wav(sys.argv[2], E, sr)
    librosa.output.write_wav(sys.argv[3], A, sr)
