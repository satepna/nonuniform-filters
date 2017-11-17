import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def decompose(b, a, dt):
    # F0 = [(b0 / dt0dt1 + b1 / dt0 + b2) / q0]
    # F1 = [(-2b0 / dt0dt1 -b1 / dt0) / q0]
    # F2 = [(b0 / dt0dt1) / q0]
    # F3 = [(2 / dt0dt1 + a1 / dt0) / q0]
    # F4 = [(1 / dt0dt1) / q0]

    (b0, b1, b2) = b
    (a0, a1, a2) = a

    dt0 = dt
    dt1 = dt
    dt0dt1 = dt0 * dt1

    q0 = (1 / dt0dt1 + a1 / dt0 + a2)

    F0 = ((b0 / dt0dt1 + b1 / dt0 + b2) / q0)
    F1 = ((-2*b0 / dt0dt1 -b1 / dt0) / q0)
    F2 = ((b0 / dt0dt1) / q0)
    F3 = ((2 / dt0dt1 + a1 / dt0) / q0)
    F4 = ((1 / dt0dt1) / q0)

    return (F0, F1, F2, F3, F4)

def recompose(F, dt0, dt1):
    # a1 = (F3 / F4 - 2) / dt1
    # a2 = [1 + (1 - F3) / F4] / dt0dt1
    # b0 = F2 / F4
    # b1 = (-F1 - 2F2) / F4 / dt1
    # b2 = (F0 + F1 + F2) / F4 / dt0dt1

    (F0, F1, F2, F3, F4) = F

    dt0dt1 = dt0 * dt1

    a0 = 1.0
    a1 = (F3 / F4 - 2) / dt1
    a2 = (1 + (1 - F3) / F4) / dt0dt1
    b0 = F2 / F4
    b1 = (-F1 - 2*F2) / F4 / dt1
    b2 = (F0 + F1 + F2) / F4 / dt0dt1

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])

def test(b, a, dt, name):
    t = np.arange(0.0, 2.0, dt)
    x = np.sin(2 * np.pi * t)
    y = sig.lfilter(b, a, x)
    plt.figure()
    plt.plot(t, x, '.-')
    plt.plot(t, y, '.-')
    plt.savefig('%s.png' % name)

def test_missing():
    freq = 50.0
    cutoff = 5.0
    dt = 1.0 / freq
    t = np.arange(0.0, 2.0, dt)
    mask = [i for i in range(len(t)) if i < 50 or i % 2 == 0]
    neg_mask = [i for i in range(len(t)) if i not in mask]
    # t = t[mask]
    x = np.sin(2 * np.pi * t)
    x[neg_mask] = 0.0

    b = sig.firwin(15, cutoff * dt)
    b2 = (cutoff * dt) * np.sinc(cutoff * dt * np.arange(-7, 7))
    a = [1.0]
    # plt.plot(b, 'o')
    # plt.plot(b2, 's')
    # plt.grid()
    # plt.show()

    # y = sig.lfilter(b2, a, x)
    y = np.zeros(np.size(x))
    inputs = []
    ts = []

    for i in range(len(x)):
        inputs.append(x[i])
        ts.append(t[i])
        prev_dt = dt

        if len(inputs) > 15:
            inputs = inputs[1:]
            prev_dt = ts[0]
            ts = ts[1:]

        filt = np.multiply(np.concatenate(([prev_dt], np.diff(ts))), cutoff * np.sinc(cutoff * (ts - ts[len(ts)/2])))
        y[i] = np.dot(inputs, filt)

    plt.figure()
    plt.plot(t[mask], x[mask], '.-', color='blue')
    plt.plot(t[mask], y[mask], '.-', color='green')
    plt.savefig('missing.png')

# sampling at 50 Hz
# then nyquist is 25 Hz
# want cutoff frequency of 10 Hz
# so 10 / 25

if __name__ == '__main__':
    test_missing()
    # make 10 Hz cutoff filter
    b, a = sig.butter(2, 10.0 / 25.0)
    # print b, a
    test(b, a, 0.02, '50hz')

    # # factor into pieces
    F = decompose(b, a, 0.02)

    # # reconstruct with the same sampling frequency
    # bnew, anew = recompose(F, 0.02, 0.02)
    # print bnew, anew

    # print ''

    # # reconstruct with 25 Hz sampling instead of 50 Hz
    # b2525, a2525 = recompose(F, 0.04, 0.04)
    b25, a25 = sig.butter(2, 10.0 / 12.5)
    # print bnew, anew
    # test(bnew, anew, 0.04, '25hz-reconstructed')

    plt.figure()

    w, h = sig.freqz(b, a)
    angles = np.unwrap(np.angle(h))

    plt.subplot(211)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.grid()

    plt.subplot(212)
    plt.plot(w, angles)
    plt.grid()

    w, h = sig.freqz(b25, a25)
    angles = np.unwrap(np.angle(h))

    plt.subplot(211)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.grid()

    plt.subplot(212)
    plt.plot(w, angles)
    plt.grid()

    plt.savefig('freqz.png')

    # # # construct filter originally at that rate and see if it matches the reconstruction
    # b, a = sig.butter(2, 10.0 / 12.5)
    # # print b, a
    # test(b, a, 0.04, '25hz')
