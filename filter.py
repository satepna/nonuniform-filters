import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# b1 = (F3 / F4 - 2) / dt1
# b0 = [1 - (1 - F3) / F4] / dt0dt1
# a2 = F2 / F4
# a1 = (-F1 - 2F2) / F4 / dt1
# a0 = (F0 + F1 + F2) / F4 / dt0dt1

# F0 = [(a2 / dt0dt1 + a1 / dt0 + a0) / q0]
# F1 = [(-2a2 / dt0dt1 -a1 / dt0) / q0]
# F2 = [(a2 / dt0dt1) / q0]
# F3 = [(2 / dt0dt1 + b1 / dt0) / q0]
# F4 = [(1 / dt0dt1) / q0]

# sampling at 50 Hz
# then nyquist is 25 Hz
# want cutoff frequency of 10 Hz
# so 10 / 25

if __name__ == '__main__':
    b, a = sig.butter(2, 10.0 / 25.0)
    print b, a
    # b is coefficients of x (input values)
    # a is coefficients of y (old output values)
    # this is swapped from my notation in my derivation :(
