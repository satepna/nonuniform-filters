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
    filters = [None] * len(t_result)

    for i in range(len(t_result)):
        mask = np.logical_and(t_input >= t_result[i] - width, t_input < t_result[i])
        t_window = t_input[mask] - (t_result[i] - width / 2.0)
        x_window = x_input[mask]

        x_result[i], filters[i] = func(width, t_window, x_window)

    return t_result, x_result, filters

def windowed_avg_func(width, t, x):
    if len(x) == 0:
        return np.nan, None

    filt = np.ones(np.size(t)) / len(t)
    return np.dot(x, filt), filt

def make_gaussian_filter(width, t):
    stddev = width / 6.75
    f = lambda p: np.exp(-(p / (2.0 * stddev))**2.0)

    filt = [f(p) for p in t]
    norm = sum(filt)

    return [p / norm for p in filt]

def windowed_gaussian_func(width, t, x):
    if len(x) == 0:
        return np.nan, None

    filt = make_gaussian_filter(width, t)
    return np.dot(filt, x), filt

if __name__ == '__main__':
    dt = 0.02
    window = 0.1

    (t, x) = make_input(dt)
    (ta, xa, filters_a) = apply_window(t, x, dt, window, windowed_avg_func)
    (tg, xg, filters_g) = apply_window(t, x, dt, window, windowed_gaussian_func)

    plt.figure()
    plt.plot(t, x, '.-', color='black')
    plt.plot(ta, xa, '.-', color='blue')
    plt.plot(tg, xg, '.-', color='red')
    plt.grid()

    plt.savefig('gaussian.png')

    plt.figure()

    for b in filters_a:
        if b is None:
            continue
        w, h = sig.freqz(b, [1.0])
        angles = np.unwrap(np.angle(h))
        plt.subplot(211)
        plt.plot(w, 20 * np.log10(abs(h)), color='blue')
        plt.subplot(212)
        plt.plot(w, angles, color='blue')

    for b in filters_g:
        if b is None:
            continue
        w, h = sig.freqz(b, [1.0])
        angles = np.unwrap(np.angle(h))
        plt.subplot(211)
        plt.plot(w, 20 * np.log10(abs(h)), color='red')
        plt.subplot(212)
        plt.plot(w, angles, color='red')

    plt.subplot(211)
    plt.ylim(-40, 0)
    plt.grid()
    plt.subplot(212)
    plt.grid()

    plt.savefig('gaussian-freqz.png')
