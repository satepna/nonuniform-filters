import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# F0 = [(b0 / dt0dt1 + b1 / dt0 + b2) / q0]
# F1 = [(-2b0 / dt0dt1 -b1 / dt0) / q0]
# F2 = [(b0 / dt0dt1) / q0]
# F3 = [(2 / dt0dt1 + a1 / dt0) / q0]
# F4 = [(1 / dt0dt1) / q0]

# a1 = (F3 / F4 - 2) / dt1
# a2 = [1 - (1 - F3) / F4] / dt0dt1
# b0 = F2 / F4
# b1 = (-F1 - 2F2) / F4 / dt1
# b2 = (F0 + F1 + F2) / F4 / dt0dt1

# sampling at 50 Hz
# then nyquist is 25 Hz
# want cutoff frequency of 10 Hz
# so 10 / 25

if __name__ == '__main__':
    b, a = sig.butter(2, 10.0 / 25.0)
    print b, a
    # b is coefficients of x (input values)
    # a is coefficients of y (old output values)

    (b0, b1, b2) = b
    (a0, a1, a2) = a

    dt0 = 0.02
    dt1 = 0.02
    dt0dt1 = dt0 * dt1

    q0 = (1 / dt0dt1 + a1 / dt0 + a2)

    F0 = ((b0 / dt0dt1 + b1 / dt0 + b2) / q0)
    F1 = ((-2*b0 / dt0dt1 -b1 / dt0) / q0)
    F2 = ((b0 / dt0dt1) / q0)
    F3 = ((2 / dt0dt1 + a1 / dt0) / q0)
    F4 = ((1 / dt0dt1) / q0)

    print F0, F1, F2, F3, F4

    a1 = (F3 / F4 - 2) / dt1
    a2 = (1 - (1 - F3) / F4) / dt0dt1
    b0 = F2 / F4
    b1 = (-F1 - 2*F2) / F4 / dt1
    b2 = (F0 + F1 + F2) / F4 / dt0dt1

    print '[', b0, b1, b2, ']', '[', 1.0, a1, a2, ']'
