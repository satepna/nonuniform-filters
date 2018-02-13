# from https://pdfs.semanticscholar.org/5078/0671847de20969fa653b689d0ce5ea05d0af.pdf
# the analytic stuff is from reference [5] in that paper, http://documents.irevues.inist.fr/bitstream/handle/2042/2173/nondispo.pdf?sequence=1

import numpy as np
import scipy.linalg as linalg

def compute_ABCD(alpha, beta):
    # pad alpha if it's too short (fewer zeros than poles)
    if len(alpha) < len(beta):
        alpha = np.pad(alpha, ((0, len(beta) - len(alpha))), 'constant')
    elif len(alpha) > len(beta):
        # I don't know how to handle this case since we need to normalize by beta[N]
        assert False

    # normalize to beta[N] = 1. the paper doesn't mention this but seems to assume it...
    assert beta[-1] != 0
    alpha = np.divide(alpha, beta[-1])
    beta = np.divide(beta, beta[-1])

    # N is the order of the filter
    assert len(alpha) == len(beta)
    N = len(beta) - 1
    assert N >= 1

    I = np.identity(N - 1)
    zeros_col = np.zeros((N - 1, 1))

    A = np.vstack((np.hstack((zeros_col, I)),
                  -beta[0:N]))

    B = np.zeros((N, 1))
    B[-1] = 1

    C = alpha[0:N] - np.multiply(alpha[N], beta[0:N])

    D = alpha[N]

    return (A, B, C, D)

def euler_step(ABCD, inputs, state, dt):
    prev_input = inputs[-2] if len(inputs) > 1 else 0
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    new_state = np.dot(I + dt * A, state) + np.dot(B * dt, prev_input)
    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    # print '-----'
    # print state
    # print new_state
    # print output

    return new_state, output

def bilinear_step(ABCD, inputs, state, dt):
    prev_input = inputs[-2] if len(inputs) > 1 else 0.0
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
    prev_input = inputs[-2] if len(inputs) > 1 else 0.0
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
    prev_input = inputs[-2] if len(inputs) > 1 else 0.0
    current_input = inputs[-1]

    (A, B, C, D) = ABCD
    I = np.identity(len(state))

    expAdt = linalg.expm(A * dt)
    invA = linalg.inv(A)
    invA2 = invA * invA

    du_dt = (current_input - prev_input) / dt

    # first order hold
    new_state = np.dot(expAdt, state) - \
                np.dot(np.dot(invA, I - expAdt), B) * prev_input + \
                np.dot(np.dot(invA2, I - np.dot(expAdt, I - A * dt)), B) * du_dt

    output = np.asscalar(np.dot(C, new_state) + np.dot(D, current_input))

    return new_state, output

def make_output(alpha, beta, nominal_dt, method, t, x):
    N = len(beta) - 1
    ABCD = compute_ABCD(alpha, beta)

    state = np.zeros(N).reshape((N, 1))
    y = np.zeros_like(x)

    for i in range(len(t)):
        new_input = x[max(0, i-1):i+1]
        state, output = method(ABCD, new_input, state, nominal_dt if i == 0 else t[i] - t[i-1])
        y[i] = output

    return y
