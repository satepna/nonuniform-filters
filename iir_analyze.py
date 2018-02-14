import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft
import matplotlib.pyplot as plt

from iir import euler_step, bilinear_step, analytic0_step, analytic1_step, make_output, prewarp, analog_zpk_to_alpha_beta


def scipy_filter(input_x, freq_bin_count, order, cutoff_freq, sample_freq, btype):
    # Construct a digital filter with the parameters above using scipy's butterworth design function.
    nyquist_freq = sample_freq / 2.0
    b_digital, a_digital = sig.butter(order, cutoff_freq / nyquist_freq, btype=btype)

    # Ask scipy to compute the response of the filter we just designed. Convert the frequency vector from digital
    # angular frequency (0 to pi) to analog cyclic frequency (0 to sample_freq) so that we can plot it together with
    # fft_freq.
    w, h = sig.freqz(b_digital, a_digital, worN=freq_bin_count)
    freqz_freq = w / np.pi * nyquist_freq
    freqz_amplitude = np.abs(h)
    freqz_phase = np.unwrap(np.angle(h))

    # Apply the digital filter that scipy gave us using scipy's lfilter function.
    filtered_output = sig.lfilter(b_digital, a_digital, input_x)

    return freqz_freq, freqz_amplitude, freqz_phase, filtered_output


def statespace_filter(method, input_t, input_x, order, cutoff_freq, sample_freq, btype):
    # Design an analog butterworth filter to our example parameters. Pre-warp the cutoff frequency since we are actually
    # going to use these zeros and poles to build a digital filter. We don'input_t actually want the analog response, even
    # though we're specifying the filter via analog parameters.
    filter_zpk = sig.butter(order, prewarp(cutoff_freq, sample_freq), output='zpk', analog=True, btype=btype)

    # Convert the filter specification to the format described in the paper, and apply the filter.
    alpha, beta = analog_zpk_to_alpha_beta(filter_zpk)
    filtered_output  = make_output(alpha, beta, 1.0 / sample_freq, method, input_t, input_x)

    return filtered_output


def analyze_filter(sample_freq, cutoff_freq, order, btype):
    # Make a chirp input to pass through our filters.
    input_t, input_x, fft_freq, input_fft = make_inputs(sample_freq)

    # Create, analyze, and apply a filter using scipy's built-in functions.
    freqz_freq, freqz_amplitude, freqz_phase, lfilter_output = scipy_filter(
        input_x, len(fft_freq), order, cutoff_freq, sample_freq, btype)

    # Create and apply a filter using this paper's method.
    statespace_output  = statespace_filter(bilinear_step, input_t, input_x, order, cutoff_freq, sample_freq, btype)

    # FFT the result of filtering the chirp by both types of filters.
    lfilter_fft = fft.fft(lfilter_output)[0:len(fft_freq)]
    statespace_fft = fft.fft(statespace_output)[0:len(fft_freq)]

    # Plot everything: input chirp, filtered chirp, and filter transfer function.
    plot_inputs('%s-' % btype, input_t, input_x, fft_freq, input_fft)

    plot_outputs('%s-' % btype, input_t, fft_freq, [
        ('lfilter', lfilter_output, lfilter_fft),
        ('statespace', statespace_output, statespace_fft),
    ])

    plot_transfer_function('%s-' % btype, freqz_freq, freqz_amplitude, freqz_phase, fft_freq, input_fft, [
        ('lfilter', lfilter_fft),
        ('statespace', statespace_fft),
    ])


def compare_methods(sample_freq, cutoff_freq, order, btype):
    # Make a chirp input to pass through our filters.
    input_t, input_x, fft_freq, input_fft = make_inputs(sample_freq)

    # Create, analyze, and apply a filter using scipy's built-in functions.
    freqz_freq, freqz_amplitude, freqz_phase, lfilter_output = scipy_filter(
        input_x, len(fft_freq), order, cutoff_freq, sample_freq, btype)

    # Create and apply a filter using this paper's method, using several integration techniques.
    outputs = {
        # 'euler':     statespace_filter(euler_step,     input_t, input_x, order, cutoff_freq, sample_freq, btype),
        'bilinear':  statespace_filter(bilinear_step,  input_t, input_x, order, cutoff_freq, sample_freq, btype),
        'analytic0': statespace_filter(analytic0_step, input_t, input_x, order, cutoff_freq, sample_freq, btype),
        'analytic1': statespace_filter(analytic1_step, input_t, input_x, order, cutoff_freq, sample_freq, btype),
    }

    # FFT the result of filtering the chirp by all types of filters.
    ffts = {
        method_name: fft.fft(output)[0:len(fft_freq)] for method_name, output in outputs.iteritems()
    }

    # Plot everything: input chirp, filtered chirp, and filter transfer function.
    plot_inputs('compare-%s-' % btype, input_t, input_x, fft_freq, input_fft)

    plot_outputs('compare-%s-' % btype, input_t, fft_freq, [
        (method_name, outputs[method_name], ffts[method_name]) for method_name in sorted(outputs.keys())
    ])

    plot_transfer_function('compare-%s-' % btype, freqz_freq, freqz_amplitude, freqz_phase, fft_freq, input_fft, [
        (method_name, ffts[method_name]) for method_name in sorted(outputs.keys())
    ])


def make_inputs(sample_freq):
    """
    Create a time vector at that sampling rate, and generate a chirp going from 0 to the Nyquist frequency across that
    time vector. We'll use this chirp to characterize our filters.
    """
    t_max = 500.0
    nyquist_freq = sample_freq / 2.0

    input_t = np.linspace(0.0, t_max, t_max * sample_freq, endpoint=False)
    input_x = sig.chirp(input_t, 0.0, t_max, nyquist_freq, phi=90.0)

    # Compute the FFT of the input so that we can compare it to the result of passing this input through various
    # filters.
    input_fft = fft.fft(input_x)
    input_fft = input_fft[0:len(input_fft)/2] # only look at positive frequency components

    fft_freq = fft.fftfreq(len(input_x), d=t_max / len(input_t))
    fft_freq = fft_freq[0:len(fft_freq) / 2]

    # Return time-domain and frequency-domain representations of the chirp.
    return input_t, input_x, fft_freq, input_fft


def plot_inputs(name_prefix, input_t, input_x, fft_freq, input_fft):
    """
    Plot this input signal and its FFT, for reference.
    """
    plt.figure()

    plt.subplot(311)
    plt.plot(input_t, input_x)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('chirp signal')

    plt.subplot(312)
    plt.plot(fft_freq, 10.0 * np.log10(np.abs(input_fft)))
    plt.grid()
    plt.xlabel('freq')
    plt.ylabel('amplitude (dB)')

    plt.subplot(313)
    plt.plot(fft_freq, np.unwrap(np.angle(input_fft)))
    plt.grid()
    plt.xlabel('freq')
    plt.ylabel('phase (rad)')

    plt.savefig('plots/%schirp.png' % name_prefix)


def plot_outputs(name_prefix, input_t, fft_freq, outputs):
    """
    Plot the result of filtering the input chirp by several filters.
    """
    plt.figure()

    plt.subplot(311)

    for name, output, output_fft in outputs:
        plt.plot(input_t, output, label=name)

    plt.grid()
    plt.xlabel('time')
    plt.ylabel('filtered chirp signal')

    plt.subplot(312)

    for name, output, output_fft in outputs:
        plt.plot(fft_freq, 10 * np.log10(np.abs(output_fft)), label=name)

    plt.grid()
    plt.xlabel('freq')
    plt.ylabel('amplitude (dB)')

    plt.subplot(313)

    for name, output, output_fft in outputs:
        plt.plot(fft_freq, np.unwrap(np.angle(output_fft)), label=name)

    plt.grid()
    plt.legend(loc='lower right')
    plt.xlabel('freq')
    plt.ylabel('phase (rad)')

    plt.savefig('plots/%schirp-filt.png' % name_prefix)


def plot_transfer_function(name_prefix, freqz_freq, freqz_amplitude, freqz_phase, fft_freq, input_fft, outputs):
    """
    Compute and plot the response of several filters by comparing the chirp input to each filter's output.
    """
    plt.figure()

    plt.subplot(211)
    plt.plot(freqz_freq, 10 * np.log10(freqz_amplitude), '.', color='black', label='freqz')

    for name, output_fft in outputs:
        plt.plot(fft_freq, 10 * np.log10(np.abs(output_fft)) - 10 * np.log10(np.abs(input_fft)), label=name)

    plt.grid()
    plt.ylabel('gain (dB)')

    plt.subplot(212)
    plt.plot(freqz_freq, freqz_phase, '.', color='black', label='freqz')

    for name, output_fft in outputs:
        plt.plot(fft_freq, np.unwrap(np.angle(output_fft) - np.angle(input_fft)), label=name)

    plt.grid()
    plt.xlabel('freq')
    plt.ylabel('phase (rad)')
    plt.legend()

    plt.savefig('plots/%sresponse.png' % name_prefix)


if __name__ == '__main__':
    # Set the parameters of the sampling system. All the filter parameters are set with respect to these, so they're
    # pretty arbitrarily chosen.
    sample_freq = 1.0

    # Set filter parameters. Arbitrarily make a second-order filter with cutoff frequency in the middle of the range.
    cutoff_freq = 0.5 * (sample_freq / 2.0)
    order = 2

    # Analyze the method in this paper compared to scipy.
    analyze_filter(sample_freq, cutoff_freq, order, 'lowpass')
    analyze_filter(sample_freq, cutoff_freq, order, 'highpass')

    # Compare methods in this paper.
    compare_methods(sample_freq, cutoff_freq, order, 'lowpass')
    compare_methods(sample_freq, cutoff_freq, order, 'highpass')
