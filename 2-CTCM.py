import numpy as np

def embedded_markov_chain(Q):
    P = np.copy(Q)
    np.fill_diagonal(P, 0)
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sums, where=row_sums!=0)  # avoid division by zero
    return P

# 0 - Example generator matrix Q
Q = np.array([
    [-1, 1, 0],
    [1, -2, 1],
    [2, 2, -4]
])

# 1 - Find Embedded Markov Transiton Probability Matrix
P = embedded_markov_chain(Q)
print("\nEmbedded Markov Chain Transition Matrix P:\n", P)

# 2 - Find Holding Parameters
def holding_parameters(Q):
    return -np.diag(Q)

holding_params = holding_parameters(Q)
print("\nHolding Parameters:\n", holding_params)

# 3 - Find the stationary distribution
def stationary_distribution(Q):
    n = Q.shape[0]
    A = np.append(Q.T, [np.ones(n)], axis = 0)
    b = np.append(np.zeros(n), 1)
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

stationary_dist = stationary_distribution(Q)
print("\nStationary Distribution:\n", stationary_dist)

# 4 - Kolmogorov Forward and Backward Equations
# These describe how the transition probabilities evolve over time
from scipy.linalg import expm
def kolmogorov_equations(Q, t):
    return expm(Q * t)

t = 1.0  # Example time
P_t = kolmogorov_equations(Q, t)
print("\nTransition probabilities at time t=1:\n", P_t)

# 5 - Expected Sojourn Time
# The expected time the process spends in each state before transitioning
def expected_sojourn_time(Q):
    return -1 / np.diag(Q)

sojourn_times = expected_sojourn_time(Q)
print("\nExpected Sojourn Times:\n", sojourn_times)

# 6 - First Passage Times
# The expected time to reach a particular state from another state for the first time
def first_passage_times(Q):
    n = Q.shape[0]
    M = np.zeros((n, n))
    for j in range(n):
        Q_j = np.delete(np.delete(Q, j, 0), j, 1)
        b = -np.ones(n-1)
        m_j = np.linalg.solve(Q_j, b)
        M[:j, j] = m_j[:j]
        M[j+1:, j] = m_j[j:]
    return M

fpt = first_passage_times(Q)
print("\nFirst Passage Times:\n", fpt)
