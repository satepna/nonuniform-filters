# from https://pdfs.semanticscholar.org/5078/0671847de20969fa653b689d0ce5ea05d0af.pdf

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def compute_ABCD(alpha, beta):
    # N is the order of the filter
    assert len(alpha) == len(beta)
    N = len(beta) - 1
    assert N >= 1

    if N == 1:
        A = np.array([[-beta[0]]])
    else:
        I = np.identity(N - 1)
        zeros_col = np.zeros((N - 1, 1))

        A = np.vstack((np.hstack((zeros_col, I)),
                      np.multiply(-1, beta[0:N])))

    B = np.zeros((N, 1))
    B[-1] = 1

    C = alpha[0:N] - np.multiply(alpha[N], beta[0:N])

    D = alpha[N]

    return (A, B, C, D)

def euler_step(ABCD, inputs, state, dt):
    (A, B, C, D) = ABCD

    prev_input = inputs[-2] if len(inputs) > 1 else 0
    current_input = inputs[-1]

    new_state = np.dot(np.identity(len(state)) + dt * A, state) + np.dot(B * dt, prev_input)
    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    print '-----'
    print state
    print new_state
    print output

    return new_state, output

def make_input(dt):
    t = np.arange(0.0, 20.0, dt)
    # mask = [i for i in range(len(t)) if i < 50 or i % 2 == 0]
    # t = t[mask]
    x = np.ones(np.size(t))#np.sin(2 * np.pi * t) + 0.05 * np.sin(123 * t)

    return (t, x)

if __name__ == '__main__':
    freq = 1000.0
    cutoff = 10.0

    # simple zero at s=-cutoff
    alpha = [1.0, 0.0]
    beta = [1.0, 1.0/cutoff]

    ABCD = compute_ABCD(alpha, beta)
    print 'A = ', ABCD[0]
    print 'B = ', ABCD[1]
    print 'C = ', ABCD[2]
    print 'D = ', ABCD[3]

    dt = 1.0 / freq
    (t, x) = make_input(dt)

    N = len(beta) - 1
    state = np.zeros(N).reshape((N, 1))
    outputs = []

    for i in range(len(t)):
        inputs = x[0:i+1]
        state, output = euler_step(ABCD, inputs, state, dt)
        # print state, output
        outputs.append(output)

    plt.plot(t, x, color='black', label='input')
    plt.plot(t, outputs, color='red', label='filtered')

    plt.grid()
    plt.legend()

    plt.savefig('iir.png')

    # plt.figure()

    # w, h = sig.freqs(*analog_filt)
    # angles = np.unwrap(np.angle(h))

    # plt.subplot(211)
    # plt.plot(w / (freq / 2), 20 * np.log10(abs(h)), color='blue', label='analog')
    # plt.grid()

    # plt.subplot(212)
    # plt.plot(w / (freq / 2), angles, color='blue', label='analog')
    # plt.grid()

    # w, h = sig.freqz(*digital_filt)
    # angles = np.unwrap(np.angle(h))

    # plt.subplot(211)
    # plt.plot(w, 20 * np.log10(abs(h)), color='red', label='digital')

    # plt.subplot(212)
    # plt.plot(w, angles, color='red', label='digital')

    # plt.legend()

    # plt.savefig('iir-freq.png')
