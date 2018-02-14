# Implementation of:
#   IIR Digital Filtering of Non-uniformly Sampled Signals via State Representation
#   L. Fesquet, B. Bidegaray-Fesquet, 2010
# Accessed from https://pdfs.semanticscholar.org/5078/0671847de20969fa653b689d0ce5ea05d0af.pdf
#
# The "analytic" methods are from reference [5] in that paper:
#   Filtering of irregularly sampled signals
#   L. FONTAINE, J. RAGOT, 2001
# Accessed from http://documents.irevues.inist.fr/bitstream/handle/2042/2173/nondispo.pdf?sequence=1

import numpy as np
import scipy.linalg as linalg


def prewarp(freq, sample_freq):
    """
    Pre-warp the passed filter frequency such that a digital filter constructed with the return value will have the same
    response as an analog filter constructed with the passed frequency.
    """
    return 2 * sample_freq * np.tan(np.pi * freq / sample_freq)


def analog_zpk_to_alpha_beta(zpk):
    """
    Convert "analog zeros, poles, gain" format returned by scipy filter design functions to analog transfer function
    polynomial coefficients in the convention used by this paper.

    The overall idea is to convert a list of roots into a list of polynomial coefficients using np.poly(), and then
    normalize the way that this paper wants.
    """
    zeros, poles, gain = zpk

    # Multiply out zeros to get transfer function numerator coefficients. np.poly() returns a bare float rather than an
    # array with 1 element if its input is empty; manually handle that case so we always get an array.
    if len(zeros) > 0:
        alpha = np.poly(zeros)
    else:
        alpha = np.array([1.0])

    # Multiply out poles to get transfer function denominator coefficients.
    if len(poles) > 0:
        beta = np.poly(poles)
    else:
        beta = np.array([1.0])

    # Reverse the order of the coefficients. np.poly() returns highest-power first, but this paper uses lowest-power
    # first.
    alpha = alpha[::-1]
    beta = beta[::-1]

    # Scale numerator by the overall system gain.
    alpha *= gain

    # Normalize so that the highest-power term in the denominator is 1.0.
    assert beta[-1] != 0.0
    alpha = np.divide(alpha, beta[-1])
    beta = np.divide(beta, beta[-1])

    return alpha, beta


def compute_ABCD(alpha, beta):
    """
    Compute the controller canonical form of a transfer function given the numerator and denominator coefficients of its
    transfer function. Note that these are the coefficients of the expanded polynomials, not the locations of the zeros
    and poles.

    See e.g. https://www.dsprelated.com/freebooks/filters/Converting_State_Space_Form_Hand.html for a derivation of this
    (although note that that site uses pretty much the opposite convention from this paper at every opportunity).
    """

    # Pad alpha if it's too short (fewer zeros than poles).
    if len(alpha) < len(beta):
        alpha = np.pad(alpha, ((0, len(beta) - len(alpha))), 'constant')
    elif len(alpha) > len(beta):
        # I don't know how to handle this case since we need to normalize by beta[N]
        assert False

    # N is the order of the filter.
    assert len(alpha) == len(beta)
    N = len(beta) - 1
    assert N >= 1

    # Normalize to beta[N] = 1. The paper doesn't mention this, but seems to assume it...
    assert beta[-1] != 0
    alpha = np.divide(alpha, beta[-1])
    beta = np.divide(beta, beta[-1])

    # Build the state transition matrix.
    I = np.identity(N - 1)
    zeros_col = np.zeros((N - 1, 1))

    A = np.vstack((np.hstack((zeros_col, I)),
                  -beta[0:N]))

    # Build the command vector.
    B = np.zeros((N, 1))
    B[-1] = 1

    # Build the observation vector.
    C = alpha[0:N] - np.multiply(alpha[N], beta[0:N])

    # Build the direct-link coefficient.
    D = alpha[N]

    # Return the state space representation of this transfer function.
    return (A, B, C, D)


def euler_step(ABCD, inputs, state, dt):
    """
    Compute one step of the filter (new state and new output) by integrating using Euler's method.
    """

    prev_input = inputs[-2]
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    new_state = np.dot(I + dt * A, state) + np.dot(B * dt, prev_input)
    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output


def bilinear_step(ABCD, inputs, state, dt):
    """
    Compute one step of the filter (new state and new output) by integrating using the bilinear method.
    """

    prev_input = inputs[-2]
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    Aminus = I - (dt/2.0) * A
    Aplus = I + (dt/2.0) * A
    Aminus_inv = linalg.inv(Aminus)

    Psi = np.dot(Aminus_inv, Aplus)
    Lambda = np.dot(Aminus_inv, B * dt)

    new_state = np.dot(Psi, state) + np.dot(Lambda, 0.5 * (prev_input + current_input))
    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output


def analytic0_step(ABCD, inputs, state, dt):
    """
    Compute one step of the filter (new state and new output) by integrating analytically, assuming that the input was
    constant over the passed dt.
    """

    prev_input = inputs[-2]
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    expAdt = linalg.expm(dt * A)
    invA = linalg.inv(A)

    # zero order hold
    new_state = np.dot(expAdt, state) - \
                np.dot(np.dot(invA, I - expAdt), B) * prev_input

    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output


def analytic1_step(ABCD, inputs, state, dt):
    """
    Compute one step of the filter (new state and new output) by integrating analytically, assuming that the input was
    linear over the passed dt.
    """

    prev_input = inputs[-2]
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    expAdt = linalg.expm(dt * A)
    invA = linalg.inv(A)
    invA2 = invA * invA

    du_dt = (current_input - prev_input) / dt

    # first order interpolation
    new_state = np.dot(expAdt, state) - \
                np.dot(np.dot(invA, I - expAdt), B) * prev_input + \
                np.dot(np.dot(invA2, I - np.dot(expAdt, I - A * dt)), B) * du_dt

    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output


def apply_filter(alpha, beta, first_dt, method, t, x):
    """
    Apply a transfer function to an input using a particular integration method (one of the *_step functions above).
    """

    # Create the state space representation of this transfer function.
    N = len(beta) - 1
    ABCD = compute_ABCD(alpha, beta)

    # Create the internal state of the system.
    state = np.zeros(N).reshape((N, 1))

    # Create the output vector.
    y = np.zeros_like(x)

    # Create a list to hold the two most recent inputs.
    new_input = [0.0, 0.0]

    # Populate the output vector by applying the step method at each sample.
    for i in range(len(t)):
        # Shift in the new input.
        new_input[0] = new_input[1]
        new_input[1] = x[i]

        # Step the filter forward by one sample.
        state, y[i] = method(ABCD, new_input, state, first_dt if i == 0 else t[i] - t[i-1])

    return y
