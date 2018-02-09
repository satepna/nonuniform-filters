# from https://pdfs.semanticscholar.org/5078/0671847de20969fa653b689d0ce5ea05d0af.pdf

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.linalg import expm

def compute_ABCD(alpha, beta):
    # pad alpha if it's too short (fewer zeros than poles)
    if len(alpha) < len(beta):
        alpha = np.pad(alpha, ((0, len(beta) - len(alpha))), 'constant')
    elif len(alpha) > len(beta):
        # I don't know how to handle this case since we need to normalize by beta[N]
        assert False

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
                  -beta[0:N]))

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

def bilinear_step(ABCD, inputs, state, dt):
    (A, B, C, D) = ABCD

    prev_input = inputs[-2] if len(inputs) > 1 else 0.0
    current_input = inputs[-1]

    Aminus = np.identity(len(state)) - (dt/2.0) * A
    Aplus = np.identity(len(state)) + (dt/2.0) * A
    Aminus_inv = np.linalg.inv(Aminus)

    Psi = np.dot(Aminus_inv, Aplus)
    Lambda = np.dot(Aminus_inv, B * dt)

    new_state = np.dot(Psi, state) + np.dot(Lambda, 0.5 * (prev_input + current_input))
    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output

def analytic0_step(ABCD, inputs, state, dt):
    (A, B, C, D) = ABCD

    prev_input = inputs[-2] if len(inputs) > 1 else 0.0
    current_input = inputs[-1]

    expA = expm(A * dt)
    invA = np.linalg.inv(A)
    I = np.identity(len(A))

    # zero order hold
    new_state = np.dot(expA, state) - \
                np.dot(np.dot(invA, I - expA), B) * prev_input

    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output

def analytic1_step(ABCD, inputs, state, dt):
    (A, B, C, D) = ABCD

    prev_input = inputs[-2] if len(inputs) > 1 else 0.0
    current_input = inputs[-1]

    expA = expm(A * dt)
    invA = np.linalg.inv(A)
    invA2 = invA * invA

    I = np.identity(len(A))
    du_dt = (current_input - prev_input) / dt

    # first order hold
    new_state = np.dot(expA, state) - \
                np.dot(np.dot(invA, I - expA), B) * prev_input + \
                np.dot(np.dot(invA2, I - np.dot(expA, I - A * dt)), B) * du_dt

    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output

def make_input(dt):
    t = np.arange(0.0, 2.0, dt)
    mask = [i for i in range(len(t)) if i < 50 or i % 2 == 0]
    t = t[mask]
    x = np.sin(2 * np.pi * t) + 0.05 * np.sin(123 * t)

    return (t, x)

def make_output(alpha, beta, nominal_dt, method, t, x):
    N = len(beta) - 1
    ABCD = compute_ABCD(alpha, beta)

    state = np.zeros(N).reshape((N, 1))
    y = np.zeros_like(x)

    for i in range(len(t)):
        new_input = x[max(0, i-1):i+1]
        state, output = method(ABCD, new_input, state, nominal_dt if i == 0 else t[i] - t[i-1])
        y[i] = output

    return y

if __name__ == '__main__':
    # TODO: need to clarify between Hz and rad/sec
    freq = 50.0
    cutoff = 10.0

    # simple zero at s=-cutoff
    alpha = [1.0]
    beta = [1.0, 1.0/cutoff]

    # 2nd order butterworth
    # alpha = [1.0]
    # beta = [1.0, np.sqrt(2)/cutoff, 1.0 / cutoff / cutoff]

    # ABCD = compute_ABCD(alpha, beta)
    # print 'A = ', ABCD[0]
    # print 'B = ', ABCD[1]
    # print 'C = ', ABCD[2]
    # print 'D = ', ABCD[3]

    # make inputs
    dt = 1.0 / freq
    (t, x) = make_input(dt)

    # filter using scipy for reference
    # need to flip the coefficients since alpha and beta are in increasing order but apparently bilinear() needs them in
    # decreasing order.
    b_digital, a_digital = sig.filter_design.bilinear(alpha[::-1], beta[::-1], freq)
    sig_outputs = sig.lfilter(b_digital, a_digital, x)

    # filter using this algorithm
    euler_outputs     = make_output(alpha, beta, dt, euler_step, t, x)
    bilinear_outputs  = make_output(alpha, beta, dt, bilinear_step, t, x)
    analytic0_outputs = make_output(alpha, beta, dt, analytic0_step, t, x)
    analytic1_outputs = make_output(alpha, beta, dt, analytic1_step, t, x)

    # plot!
    plt.plot(t, x, '.-', color='black', label='input')
    plt.plot(t, sig_outputs, '.-', color='gray', label='lfilter')
    plt.plot(t, euler_outputs, '.-', color='green', label='euler')
    plt.plot(t, bilinear_outputs, '.-', color='blue', label='bilinear')
    plt.plot(t, analytic0_outputs, '.-', color='red', label='analytic0')
    plt.plot(t, analytic1_outputs, '.-', color='pink', label='analytic1')

    plt.grid()
    plt.legend()

    plt.savefig('iir.png')

    plt.figure()

    # Plot frequency response of this filter that we made, for reference.
    w, h = sig.freqz(b_digital, a_digital)
    angles = np.unwrap(np.angle(h))

    plt.subplot(211)
    plt.plot(w / np.pi * freq, 20 * np.log10(abs(h)), color='red')
    plt.axvline(cutoff, color='black')
    plt.grid()

    plt.subplot(212)
    plt.plot(w / np.pi * freq, angles, color='red')
    plt.axvline(cutoff, color='black')
    plt.grid()

    plt.savefig('iir-freq.png')
