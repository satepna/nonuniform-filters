import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft
import matplotlib.pyplot as plt

from iir import prewarp, analog_zpk_to_alpha_beta, bilinear_step, apply_filter


def make_input(signal_freq, dt):
    # Make a time vector that starts at dt, but drops to dt/5 after the halfway mark.
    t = np.arange(0.0, 2.0, dt)
    mask = [i for i in range(len(t)) if i < 50 or i % 5 == 0]
    t = t[mask]

    # Make a noisy sine wave basedo n that time vector.
    x = np.sin(signal_freq * 2 * np.pi * t) + 0.05 * np.sin(123 * t)

    return (t, x)


def make_timedomain_plots():
    # Design our lowpass filter.
    sample_freq = 50.0
    cutoff_freq = 5.0
    alpha, beta = analog_zpk_to_alpha_beta(sig.butter(2, prewarp(cutoff_freq, sample_freq), output='zpk', analog=True))

    # Make our example input.
    signal_freq = 1.0
    dt = 1.0 / sample_freq
    (t, x) = make_input(signal_freq, dt)

    # Filter using scipy for reference.
    b_digital, a_digital = sig.butter(2, cutoff_freq / sample_freq)
    lfilter_output = sig.lfilter(b_digital, a_digital, x)

    # Filter using the algorithm from this paper.
    statespace_output = apply_filter(alpha, beta, dt, bilinear_step, t, x)

    # Compute the frequency response of the filter we designed.
    w, h = sig.freqz(b_digital, a_digital)
    response_freq = w / (2 * np.pi) * sample_freq
    response_amplitude = np.abs(h)
    response_phase = np.unwrap(np.angle(h))
    response_group_delay = -np.diff(response_phase) / np.diff(response_freq) * dt

    # Compute the expected gain, phase, and group delay given the signal that we're providing as input, so we can mark
    # them on the graph.
    expected_gain = np.interp(signal_freq, response_freq, response_amplitude)
    expected_phase = np.interp(signal_freq, response_freq, response_phase)
    expected_group_delay = np.interp(signal_freq, response_freq[1:], response_group_delay)

    # Plot the frequency response.
    plt.figure()

    plt.subplot(311)
    plt.plot(response_freq, 10 * np.log10(response_amplitude), color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [10 * np.log10(expected_gain)], facecolors='none', edgecolors='red')
    plt.xlim(0, sample_freq / 2)
    plt.ylim(-40, 0)
    plt.ylabel('gain (dB)')
    plt.grid()

    plt.subplot(312)
    plt.plot(response_freq, response_phase, color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [expected_phase], facecolors='none', edgecolors='red')
    plt.xlim(0, sample_freq / 2)
    plt.ylim(-np.pi, 0)
    plt.ylabel('phase (rad)')
    plt.grid()

    plt.subplot(313)
    plt.plot(response_freq[1:], response_group_delay, color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [expected_group_delay], facecolors='none', edgecolors='red')
    plt.xlim(0, sample_freq / 2)
    plt.xlabel('freq (Hz)')
    plt.ylabel('group delay (s)')
    plt.grid()

    plt.savefig('plots/example-filter-design.png')

    # Plot the time-domain response of this signal, with guidelines for expected gain.
    plt.figure()
    plt.axhline(y=expected_gain, color='gray')
    plt.axhline(y=-expected_gain, color='gray')
    plt.plot(t, x, '.-', color='black', label='input')
    plt.plot(t, lfilter_output, '.-', color='gray', label='lfilter')
    plt.plot(t, statespace_output, '.-', color='red', label='statespace')

    plt.ylim(-1.5, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('signal value')
    plt.grid()
    plt.legend()

    plt.savefig('plots/example-timedomain-output.png')


if __name__ == '__main__':
    make_timedomain_plots()
