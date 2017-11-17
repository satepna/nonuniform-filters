import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def make_input(dt):
    t = np.arange(0.0, 2.0, dt)
    mask = [i for i in range(len(t)) if i < 50 or i % 2 == 0]

    t = t[mask]
    x = np.sin(2 * np.pi * t) + 0.05 * np.sin(123 * t)

    return (t, x)

def apply_window(t_input, x_input, dt, width, func):
    t_result = np.arange(0.0, 2.0, dt)
    x_result = np.zeros(np.size(t_result))

    for i in range(len(t_result)):
        mask = np.logical_and(t_input >= t_result[i] - width, t_input < t_result[i])
        t_window = t_input[mask] - (t_result[i] - width / 2.0)
        x_window = x_input[mask]

        x_result[i] = func(width, t_window, x_window)

    return t_result, x_result

def windowed_avg_func(width, t, x):
    return np.average(x) if len(x) > 0 else np.nan

def make_gaussian_filter(width, t):
    stddev = width / 6.75
    f = lambda p: np.exp(-(p / (2.0 * stddev))**2.0)

    filt = [f(p) for p in t]
    norm = sum(filt)

    return [p / norm for p in filt]

def windowed_gaussian_func(width, t, x):
    if len(x) == 0:
        return np.nan

    filt = make_gaussian_filter(width, t)
    return np.dot(filt, x)

if __name__ == '__main__':
    dt = 0.02
    window = 0.1

    (t, x) = make_input(dt)
    (ta, xa) = apply_window(t, x, dt, window, windowed_avg_func)
    (tg, xg) = apply_window(t, x, dt, window, windowed_gaussian_func)

    plt.figure()
    plt.plot(t, x, '.-', color='black')
    plt.plot(ta, xa, '.-', color='blue')
    plt.plot(tg, xg, '.-', color='red')
    plt.grid()

    plt.savefig('gaussian.png')

    plt.figure()

    ba = [0.2, 0.2, 0.2, 0.2, 0.2]
    aa = [1.0]

    bg = make_gaussian_filter(window, [-0.04, -0.02, 0.0, 0.02, 0.04])
    ag = [1.0]

    wa, ha = sig.freqz(ba, aa)
    angles_a = np.unwrap(np.angle(ha))

    wg, hg = sig.freqz(bg, ag)
    angles_g = np.unwrap(np.angle(hg))

    plt.subplot(211)
    plt.plot(wa, 20 * np.log10(abs(ha)), color='blue')
    plt.plot(wg, 20 * np.log10(abs(hg)), color='red')
    plt.grid()

    plt.subplot(212)
    plt.plot(wa, angles_a, color='blue')
    plt.plot(wg, angles_g, color='red')
    plt.grid()

    plt.savefig('gaussian-freqz.png')
