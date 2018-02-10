# from https://pdfs.semanticscholar.org/5078/0671847de20969fa653b689d0ce5ea05d0af.pdf
# the analytic stuff is from reference [5] in that paper, http://documents.irevues.inist.fr/bitstream/handle/2042/2173/nondispo.pdf?sequence=1

import numpy as np
import scipy.signal as sig
import scipy.linalg as linalg
import scipy.fftpack as fft
import matplotlib.pyplot as plt

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
    prev_input = inputs[-2] if len(inputs) > 1 else 0
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    new_state = np.dot(I + dt * A, state) + np.dot(B * dt, prev_input)
    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    # print '-----'
    # print state
    # print new_state
    # print output

    return new_state, output

def bilinear_step(ABCD, inputs, state, dt):
    prev_input = inputs[-2] if len(inputs) > 1 else 0.0
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    Aminus = I - (dt/2.0) * A
    Aplus = I + (dt/2.0) * A
    Aminus_inv = linalg.inv(Aminus)

    Psi = np.dot(Aminus_inv, Aplus)
    Lambda = np.dot(Aminus_inv, B * dt)

    new_state = np.dot(Psi, state) + np.dot(Lambda, 0.5 * (prev_input + current_input))
    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output

def analytic0_step(ABCD, inputs, state, dt):
    prev_input = inputs[-2] if len(inputs) > 1 else 0.0
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    expAdt = linalg.expm(dt * A)
    invA = linalg.inv(A)

    # zero order hold
    new_state = np.dot(expAdt, state) - \
                np.dot(np.dot(invA, I - expAdt), B) * prev_input

    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output

def analytic1_step(ABCD, inputs, state, dt):
    prev_input = inputs[-2] if len(inputs) > 1 else 0.0
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    expAdt = linalg.expm(A * dt)
    invA = linalg.inv(A)
    invA2 = invA * invA

    du_dt = (current_input - prev_input) / dt

    # first order hold
    new_state = np.dot(expAdt, state) - \
                np.dot(np.dot(invA, I - expAdt), B) * prev_input + \
                np.dot(np.dot(invA2, I - np.dot(expAdt, I - A * dt)), B) * du_dt

    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output

def make_input(signal_freq, dt):
    t = np.arange(0.0, 2.0, dt)
    mask = [i for i in range(len(t)) if i < 50 or i % 2 == 0]
    t = t[mask]
    x = np.sin(signal_freq * 2 * np.pi * t) + 0.05 * np.sin(123 * t)

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

def make_timedomain_plots():
    # TODO: need to clarify between Hz and rad/sec
    freq = 50.0
    cutoff_freq = 10.0

    # simple zero at s=-cutoff_freq
    alpha = [1.0]
    beta = [1.0, 1.0/cutoff_freq]

    # 2nd order butterworth
    # alpha = [1.0]
    # beta = [1.0, np.sqrt(2)/cutoff_freq, 1.0 / cutoff_freq / cutoff_freq]

    # ABCD = compute_ABCD(alpha, beta)
    # print 'A = ', ABCD[0]
    # print 'B = ', ABCD[1]
    # print 'C = ', ABCD[2]
    # print 'D = ', ABCD[3]

    # make inputs
    signal_freq = 1.0
    dt = 1.0 / freq
    (t, x) = make_input(signal_freq, dt)

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

    ####

    # Plot frequency response of this filter that we made.
    w, h = sig.freqz(b_digital, a_digital)
    response_freq = w / (2 * np.pi) * freq
    response_phase = np.unwrap(np.angle(h))
    response_amplitude = np.abs(h)

    expected_gain = np.interp(signal_freq, response_freq, response_amplitude)
    expected_phase = np.interp(signal_freq, response_freq, response_phase)

    plt.figure()
    plt.subplot(211)
    plt.plot(response_freq, 20 * np.log10(response_amplitude), color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [20 * np.log10(expected_gain)], facecolors='none', edgecolors='red')
    plt.xlim(0, freq / 2)
    plt.ylim(-60, 0)
    plt.grid()

    plt.subplot(212)
    plt.plot(response_freq, response_phase * 180 / np.pi, color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [expected_phase * 180 / np.pi], facecolors='none', edgecolors='red')
    plt.xlim(0, freq / 2)
    plt.ylim(-90, 0)
    plt.grid()

    plt.tight_layout()
    plt.savefig('iir-freq.png')

    ###

    # plot example signal
    plt.figure(figsize=(10, 8))
    plt.axhline(y=expected_gain, color='gray')
    plt.axhline(y=-expected_gain, color='gray')
    plt.plot(t, x, '.-', color='black', label='input')
    plt.plot(t, sig_outputs, '.-', color='gray', label='lfilter')
    plt.plot(t, euler_outputs, '.-', color='green', label='euler')
    plt.plot(t, bilinear_outputs, '.-', color='blue', label='bilinear')
    plt.plot(t, analytic0_outputs, '.-', color='red', label='analytic0')
    plt.plot(t, analytic1_outputs, '.-', color='pink', label='analytic1')

    plt.ylim(-1.5, 1.5)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig('iir.png')

def make_freqdomain_plots():
    nyquist_freq = 1.0
    sample_freq = 2.0 * nyquist_freq
    cutoff_freq = 0.1 * nyquist_freq

    # make inputs
    t_max = 4000.0
    t = np.linspace(0.0, t_max, t_max * sample_freq, endpoint=False)
    x = sig.chirp(t, 0.0, t_max, nyquist_freq)

    print t
    print x

    # plot fft of the chirp for reference
    y = fft.fft(x)
    y = y[0:len(y)/2] # only look at positive frequency components

    plt.figure()
    plt.plot(t, x)
    plt.grid()
    plt.savefig('chirp.png')

    plt.figure()

    f = fft.fftfreq(len(x), d=1.0 / sample_freq)
    f = f[0:len(f) / 2]

    plt.subplot(211)
    plt.plot(f, np.abs(y))
    plt.grid()

    plt.subplot(212)
    plt.plot(f, np.unwrap(np.angle(y)))
    plt.grid()

    plt.savefig('chirp-fft.png')

    # simple zero at s=-cutoff_freq
    alpha = [1.0]
    beta = [1.0, 1.0/cutoff_freq]

    # filter chirp using scipy.
    # need to flip the coefficients since alpha and beta are in increasing order but apparently bilinear() needs them in
    # decreasing order.
    b_digital, a_digital = sig.filter_design.bilinear(alpha[::-1], beta[::-1], sample_freq)
    sig_outputs = sig.lfilter(b_digital, a_digital, x)

    w, h = sig.freqz(b_digital, a_digital)
    expected_freq = w / np.pi
    expected_phase = np.unwrap(np.angle(h))
    expected_amplitude = np.abs(h)

    # fft the filtered chirp
    filtered_y = fft.fft(sig_outputs)
    filtered_y = filtered_y[0:len(filtered_y)/2]

    plt.figure()
    plt.subplot(211)
    plt.plot(f, np.abs(filtered_y))
    plt.grid()

    plt.subplot(212)
    plt.plot(f, np.unwrap(np.angle(filtered_y)))
    plt.grid()
    plt.savefig('chirp-filt-fft.png')

    # plot the difference of the original chirp spectrum and the filtered chirp spectrum
    plt.figure()
    plt.subplot(211)
    plt.plot(f, 20 * np.log10(np.abs(filtered_y)) - 20 * np.log10(np.abs(y)))
    plt.plot(expected_freq, 20 * np.log10(expected_amplitude), color='red')
    plt.grid()

    plt.subplot(212)
    plt.plot(f, np.unwrap(np.angle(filtered_y) - np.angle(y)))
    plt.plot(expected_freq, expected_phase, color='red')
    plt.grid()
    plt.savefig('chirp-filt-diff-fft.png')

if __name__ == '__main__':
    # make_timedomain_plots()
    make_freqdomain_plots()
