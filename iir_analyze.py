import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft
import matplotlib.pyplot as plt

from iir import euler_step, bilinear_step, analytic0_step, analytic1_step, make_output, prewarp, analog_zpk_to_alpha_beta

def make_freqdomain_plots():
    nyquist_freq = 1.0
    sample_freq = 2.0 * nyquist_freq
    cutoff_freq = 0.5 * nyquist_freq
    order = 1

    # make inputs
    t_max = 4000.0
    t = np.linspace(0.0, t_max, t_max * sample_freq, endpoint=False)
    x = sig.chirp(t, 0.0, t_max, nyquist_freq)

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
    # alpha = [1.0]
    # beta = [1.0, 1.0/cutoff_freq]
    # pre-warp frequency since we are going to use this filter info to build a digital filter.
    butter_zpk = sig.butter(order, prewarp(cutoff_freq, sample_freq), output='zpk', analog=True)
    alpha, beta = analog_zpk_to_alpha_beta(butter_zpk)

    # print '---'
    # print butter_zpk
    # print analog_zpk_to_alpha_beta(butter_zpk)
    # print alpha, beta
    # print '---'

    # filter chirp using scipy.
    # need to flip the coefficients since alpha and beta are in increasing order but apparently bilinear() needs them in
    # decreasing order.
    #b_digital, a_digital = sig.filter_design.bilinear(alpha[::-1], beta[::-1], sample_freq)
    b_digital, a_digital = sig.butter(order, cutoff_freq, output='ba', analog=False)
    lfilter_outputs = sig.lfilter(b_digital, a_digital, x)

    w, h = sig.freqz(b_digital, a_digital, worN=len(f))
    expected_freq = w / np.pi
    expected_phase = np.unwrap(np.angle(h))
    expected_amplitude = np.abs(h)

    # filter with this paper's method
    euler_outputs  = make_output(alpha, beta, 1.0 / sample_freq, euler_step, t, x)
    bilinear_outputs  = make_output(alpha, beta, 1.0 / sample_freq, bilinear_step, t, x)
    analytic0_outputs  = make_output(alpha, beta, 1.0 / sample_freq, analytic0_step, t, x)
    analytic1_outputs  = make_output(alpha, beta, 1.0 / sample_freq, analytic1_step, t, x)

    # fft the filtered chirp
    filtered_y = fft.fft(lfilter_outputs)
    filtered_y = filtered_y[0:len(filtered_y)/2]

    filtered_euler = fft.fft(euler_outputs)
    filtered_euler = filtered_euler[0:len(filtered_euler)/2]

    filtered_bilinear = fft.fft(bilinear_outputs)
    filtered_bilinear = filtered_bilinear[0:len(filtered_bilinear)/2]

    filtered_analytic0 = fft.fft(analytic0_outputs)
    filtered_analytic0 = filtered_analytic0[0:len(filtered_analytic0)/2]

    filtered_analytic1 = fft.fft(analytic1_outputs)
    filtered_analytic1 = filtered_analytic1[0:len(filtered_analytic1)/2]

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
    plt.plot(f, 10 * np.log10(np.abs(filtered_y)) - 10 * np.log10(np.abs(y)))
    # plt.plot(f, 10 * np.log10(np.abs(filtered_euler)) - 10 * np.log10(np.abs(y)))
    plt.plot(f, 10 * np.log10(np.abs(filtered_bilinear)) - 10 * np.log10(np.abs(y)))
    # plt.plot(f, 10 * np.log10(np.abs(filtered_analytic0)) - 10 * np.log10(np.abs(y)))
    # plt.plot(f, 10 * np.log10(np.abs(filtered_analytic1)) - 10 * np.log10(np.abs(y)))
    # plt.plot(f, 10 * np.log10(np.abs(filtered_y)))
    plt.plot(expected_freq, 10 * np.log10(expected_amplitude), color='black')
    plt.grid()

    plt.subplot(212)
    plt.plot(f, np.unwrap(np.angle(filtered_y)) - np.unwrap(np.angle(y)), label='lfilter')
    # plt.plot(f, np.unwrap(np.angle(filtered_euler)) - np.unwrap(np.angle(y)), label='euler')
    plt.plot(f, np.unwrap(np.angle(filtered_bilinear)) - np.unwrap(np.angle(y)), label='bilinear')
    # plt.plot(f, np.unwrap(np.angle(filtered_analytic0)) - np.unwrap(np.angle(y)), label='analytic0')
    # plt.plot(f, np.unwrap(np.angle(filtered_analytic1)) - np.unwrap(np.angle(y)), label='analytic1')
    # plt.plot(f, np.unwrap(np.angle(filtered_y)))
    plt.plot(expected_freq, expected_phase, color='black', label='freqz')
    plt.ylim(-np.pi, 0)
    plt.grid()
    plt.legend()

    plt.savefig('chirp-filt-diff-fft.png')

    pow_ref = np.sum(np.conj(h) * h) / np.max(np.abs(h))**2
    projection_lfilter = np.sum(np.conj(filtered_y / y) * h) / (np.max(np.abs(filtered_y / y)) * np.max(np.abs(h)))
    projection_euler = np.sum(np.conj(filtered_euler / y) * h) / (np.max(np.abs(filtered_euler / y)) * np.max(np.abs(h)))
    projection_bilinear = np.sum(np.conj(filtered_bilinear / y) * h) / (np.max(np.abs(filtered_bilinear / y)) * np.max(np.abs(h)))
    projection_analytic0 = np.sum(np.conj(filtered_analytic0 / y) * h) / (np.max(np.abs(filtered_analytic0 / y)) * np.max(np.abs(h)))
    projection_analytic1 = np.sum(np.conj(filtered_analytic1 / y) * h) / (np.max(np.abs(filtered_analytic1 / y)) * np.max(np.abs(h)))

    print np.abs(pow_ref), np.abs(projection_lfilter), np.abs(projection_euler), np.abs(projection_bilinear), np.abs(projection_analytic0), np.abs(projection_analytic1)

    plt.figure()
    plt.plot(t, lfilter_outputs, 'o')
    # plt.plot(t, euler_outputs)
    plt.plot(t, bilinear_outputs)
    # plt.plot(t, analytic0_outputs)
    # plt.plot(t, analytic1_outputs)
    plt.savefig('chirp-outputs.png')
    # plt.show()

if __name__ == '__main__':
    make_freqdomain_plots()
