import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft
import matplotlib.pyplot as plt

from iir import prewarp, analog_zpk_to_alpha_beta, bilinear_step, apply_filter


def make_noisy_signal_plot():
    # Make a time vector that starts at dt, but drops to dt/5 after the halfway mark.
    sample_freq = 50.0
    dt = 1.0 / sample_freq
    signal_freq = 1.0

    t = np.arange(-1.0, 2.0, dt)
    mask = [i for i in xrange(len(t)) if t[i] <= 0 or i % 5 == 0]
    t = t[mask]

    # Make a noisy sine wave based on that time vector.
    x = np.sin(signal_freq * 2 * np.pi * t) + 0.05 * np.sin(123 * t)

    # Design our lowpass filter.
    nyquist_freq = sample_freq / 2.0
    cutoff_freq = 5.0
    alpha, beta = analog_zpk_to_alpha_beta(sig.butter(2, prewarp(cutoff_freq, sample_freq), output='zpk', analog=True))

    # Filter using scipy for reference.
    b_digital, a_digital = sig.butter(2, cutoff_freq / nyquist_freq)
    lfilter_output = sig.lfilter(b_digital, a_digital, x)

    # Filter using the algorithm from this paper.
    statespace_output = apply_filter(alpha, beta, bilinear_step, t, x)

    # Compute the frequency response of the filter we designed.
    w, h = sig.freqz(b_digital, a_digital)
    response_freq = w / (2 * np.pi) * sample_freq
    response_amplitude = np.abs(h)
    response_phase = np.unwrap(np.angle(h))

    # Compute the expected gain and phase given the signal that we're providing as input, so we can mark them on the
    # graph.
    expected_gain = np.interp(signal_freq, response_freq, response_amplitude)
    expected_phase = np.interp(signal_freq, response_freq, response_phase)

    # Plot the frequency response.
    plt.figure()

    plt.subplot(211)
    plt.plot(response_freq, 10 * np.log10(response_amplitude), color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [10 * np.log10(expected_gain)], facecolors='none', edgecolors='red')
    plt.xlim(0, sample_freq / 2)
    plt.ylim(-40, 0)
    plt.ylabel('gain (dB)')
    plt.grid()

    plt.subplot(212)
    plt.plot(response_freq, response_phase, color='red')
    plt.axvline(cutoff_freq, color='black')
    plt.scatter([signal_freq], [expected_phase], facecolors='none', edgecolors='red')
    plt.xlim(0, sample_freq / 2)
    plt.ylim(-np.pi, 0)
    plt.xlabel('freq (Hz)')
    plt.ylabel('phase (rad)')
    plt.grid()

    plt.savefig('plots/example-filter-design.png')

    # Plot the time-domain response of this signal.
    plt.figure()
    plt.plot(t, x, '.-', color='black', label='input')
    plt.plot(t, lfilter_output, '.-', color='gray', label='lfilter')
    plt.plot(t, statespace_output, '.-', color='red', label='statespace')

    plt.ylim(-1.5, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('signal value')
    plt.grid()
    plt.legend()

    plt.savefig('plots/example-timedomain-noisy-signal.png')


def make_noisy_timevector_plot():
    # Make a time vector with irregular spacing.
    sample_freq = 50.0
    dt = 1.0 / sample_freq
    signal_freq = 1.0

    t = np.arange(0.0, 2.0, dt)
    t = t + 0.5 * dt * np.sin(123 * t)
    assert all(np.diff(t) > 0.0)

    # Make a pure sine wave based on that time vector.
    x = np.sin(signal_freq * 2 * np.pi * t)

    # Design our lowpass filter.
    nyquist_freq = sample_freq / 2.0
    cutoff_freq = 5.0
    alpha, beta = analog_zpk_to_alpha_beta(sig.butter(2, prewarp(cutoff_freq, sample_freq), output='zpk', analog=True))

    # Filter using scipy for reference.
    b_digital, a_digital = sig.butter(2, cutoff_freq / nyquist_freq)
    lfilter_output = sig.lfilter(b_digital, a_digital, x)

    # Filter using the algorithm from this paper.
    statespace_output = apply_filter(alpha, beta, bilinear_step, t, x)

    # Plot the time-domain response of this signal.
    plt.figure()
    plt.subplot(211)
    plt.plot(t, x, '.-', color='black', label='input')
    plt.plot(t, lfilter_output, '.-', color='gray', label='lfilter')
    plt.grid()
    plt.ylabel('signal value')
    plt.legend()

    plt.subplot(212)
    plt.plot(t, x, '.-', color='black', label='input')
    plt.plot(t, statespace_output, '.-', color='red', label='statespace')

    plt.xlabel('time (s)')
    plt.ylabel('signal value')
    plt.grid()
    plt.legend()

    plt.savefig('plots/example-timedomain-noisy-timevector.png')


if __name__ == '__main__':
    make_noisy_signal_plot()
    make_noisy_timevector_plot()
