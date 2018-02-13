import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft
import matplotlib.pyplot as plt

from iir import compute_ABCD, euler_step, bilinear_step, analytic0_step, analytic1_step, make_output

def make_input(signal_freq, dt):
    t = np.arange(0.0, 2.0, dt)
    mask = [i for i in range(len(t)) if i < 50 or i % 2 == 0]
    t = t[mask]
    x = np.sin(signal_freq * 2 * np.pi * t) + 0.05 * np.sin(123 * t)

    return (t, x)

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
    lfilter_outputs = sig.lfilter(b_digital, a_digital, x)

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
    response_group_delay = -np.diff(response_phase) / np.diff(response_freq)
    response_amplitude = np.abs(h)

    expected_gain = np.interp(signal_freq, response_freq, response_amplitude)
    expected_phase = np.interp(signal_freq, response_freq, response_phase)
    expected_group_delay = np.interp(signal_freq, response_freq[1:], response_group_delay)

    plt.figure()
    plt.subplot(311)
    plt.plot(response_freq, 20 * np.log10(response_amplitude), color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [20 * np.log10(expected_gain)], facecolors='none', edgecolors='red')
    plt.xlim(0, freq / 2)
    plt.ylim(-60, 0)
    plt.grid()

    plt.subplot(312)
    plt.plot(response_freq, response_phase * 180 / np.pi, color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [expected_phase * 180 / np.pi], facecolors='none', edgecolors='red')
    plt.xlim(0, freq / 2)
    plt.ylim(-90, 0)
    plt.grid()

    plt.subplot(313)
    plt.plot(response_freq[1:], response_group_delay, color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [expected_group_delay], facecolors='none', edgecolors='red')
    plt.xlim(0, freq / 2)
    plt.ylim(0.0, 1.0)
    plt.grid()

    plt.savefig('iir-freq.png')

    ###

    # plot example signal
    plt.figure(figsize=(10, 8))
    plt.axhline(y=expected_gain, color='gray')
    plt.axhline(y=-expected_gain, color='gray')
    plt.plot(t, x, '.-', color='black', label='input')
    plt.plot(t, lfilter_outputs, '.-', color='gray', label='lfilter')
    plt.plot(t, euler_outputs, '.-', color='green', label='euler')
    plt.plot(t, bilinear_outputs, '.-', color='blue', label='bilinear')
    plt.plot(t, analytic0_outputs, '.-', color='red', label='analytic0')
    plt.plot(t, analytic1_outputs, '.-', color='pink', label='analytic1')

    plt.ylim(-1.5, 1.5)
    plt.grid()
    plt.legend()

    plt.savefig('iir.png')

if __name__ == '__main__':
    make_timedomain_plots()
