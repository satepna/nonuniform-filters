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
    t = np.arange(0.0, 2.0, 0.02)
    mask = [i for i in range(len(t)) if i < 50 or i % 2 == 0]
    t = t[mask]
    # t = np.delete(t, 50)
    # t = np.delete(t, 60)
    x = np.sin(2 * np.pi * t)

    b50, a50 = sig.butter(2, 10.0 / 25.0)
    b25, a25 = sig.butter(2, 10.0 / 12.5)
    # b25, a25 = sig.butter(2, 10.0 / 17)
    # F = decompose(b50, a50, 0.02)
    # b25, a25 = recompose(F, 0.04, 0.02)

    y = np.zeros(np.size(x))
    for i in range(len(x)):
        if i > 1:
            dt0 = t[i] - t[i-1]
            # dt1 = t[i-1] - t[i-2]

            if dt0 > 0.025:
                b, a = b25, a25
            else:
                b, a = b50, a50

            # # if dt0 > 0.025 or dt1 > 0.025:
            # b, a = recompose(F, dt0, dt1)

            # b, a = b25, a25
        else:
            b, a = b50, a50

        xi = x[i]
        xp = x[i-1] if i-1 >= 0 else 0.0
        xpp = x[i-2] if i-2 >= 0 else 0.0

        yp = y[i-1] if i-1 >= 0 else 0.0
        ypp = y[i-2] if i-2 >= 0 else 0.0

        yi = b[0] * xi + b[1] * xp + b[2] * xpp - a[1] * yp - a[2] * ypp
        y[i] = yi

    # y = sig.lfilter(b25, a25, x)

    plt.figure()
    plt.plot(t, x, '.-')
    plt.plot(t, y, '.-')
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
