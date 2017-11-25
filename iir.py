# from https://pdfs.semanticscholar.org/5078/0671847de20969fa653b689d0ce5ea05d0af.pdf

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def compute_ABCD(alpha, beta):
    # normalize to beta[N] = 1. the paper doesn't mention this but seems to assume it...
    assert beta[-1] != 0
    alpha = np.divide(alpha, beta[-1])
    beta = np.divide(beta, beta[-1])

    # N is the order of the filter
    assert len(alpha) == len(beta)
    N = len(beta) - 1
    assert N >= 1

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

    # print '-----'
    # print state
    # print new_state
    # print output

    return new_state, output

def make_input(dt):
    t = np.arange(0.0, 2.0, dt)
    mask = [i for i in range(len(t)) if i < 50 or i % 2 == 0]
    t = t[mask]
    x = np.sin(2 * np.pi * t) + 0.05 * np.sin(123 * t)

    return (t, x)

if __name__ == '__main__':
    freq = 50.0
    cutoff = 10.0

    # simple zero at s=-cutoff
    alpha = [1.0, 0.0]
    beta = [1.0, 1.0/cutoff]

    ABCD = compute_ABCD(alpha, beta)
    print 'A = ', ABCD[0]
    print 'B = ', ABCD[1]
    print 'C = ', ABCD[2]
    print 'D = ', ABCD[3]

    # make inputs
    dt = 1.0 / freq
    (t, x) = make_input(dt)

    # filter using this algorithm
    N = len(beta) - 1
    state = np.zeros(N).reshape((N, 1))
    outputs = []

    for i in range(len(t)):
        inputs = x[0:i+1]
        state, output = euler_step(ABCD, inputs, state, dt if i == 0 else t[i] - t[i-1])
        # print state, output
        outputs.append(output)

    # filter using scipy for reference
    # zpk_digital = sig.filter_design._zpkbilinear([], [-cutoff], 1.0, freq)
    # print zpk_digital

    # assert len(zpk_digital[0]) == 1
    # assert len(zpk_digital[1]) == 1

    # b_digital = np.concatenate(([1.0], -zpk_digital[0]))
    # a_digital = np.concatenate(([1.0], -zpk_digital[1]))

    # This is what we're actually doing here in practice :(
    # b_digital = [0.01]
    # a_digital = [1.0, -0.99]

    # This is what I want to be doing based on https://en.wikipedia.org/wiki/Bilinear_transform#Example:
    b_digital = [1.0, 1.0]
    a_digital = [1 + 2/cutoff * freq, 1 - 2/cutoff * freq]

    print b_digital, a_digital
    sig_outputs = sig.lfilter(b_digital, a_digital, x)

    # butter = sig.butter(1, cutoff / freq)
    # print butter
    # butter_outputs = sig.lfilter(butter[0], butter[1], x)

    # plot!
    plt.plot(t, x, '.-', color='black', label='input')
    plt.plot(t, sig_outputs, '.-', color='blue', label='lfilter')
    # plt.plot(t, butter_outputs, color='green', label='butter')
    plt.plot(t, outputs, '.-', color='red', label='this paper')

    plt.grid()
    plt.legend()

    plt.savefig('iir.png')

    # plt.figure()

    w, h = sig.freqz(b_digital, a_digital)
    angles = np.unwrap(np.angle(h))

    plt.subplot(211)
    plt.plot(w / np.pi * freq, 20 * np.log10(abs(h)), color='red')
    plt.grid()

    plt.subplot(212)
    plt.plot(w / np.pi * freq, angles, color='red')
    plt.grid()

    plt.savefig('iir-freq.png')
