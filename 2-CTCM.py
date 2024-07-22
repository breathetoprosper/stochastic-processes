import numpy as np

def embedded_markov_chain(Q):
    P = np.copy(Q)
    np.fill_diagonal(P, 0)
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sums, where=row_sums!=0)  # avoid division by zero
    return P

# Example generator matrix Q
Q = np.array([
    [-1, 1, 0],
    [1, -2, 1],
    [2, 2, -4]
])

# Find Embedded Markov Transiton Probability Matrix
P = embedded_markov_chain(Q)
print("\nEmbedded Markov Chain Transition Matrix P:\n", P)

# Find Holding Parameters
def holding_parameters(Q):
    return -np.diag(Q)

holding_params = holding_parameters(Q)
print("\nHolding Parameters:\n", holding_params)

# Find the stationary distribution
def stationary_distribution(Q):
    n = Q.shape[0]
    A = np.append(Q.T, [np.ones(n)], axis = 0)
    b = np.append(np.zeros(n), 1)
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

stationary_dist = stationary_distribution(Q)
print("\nStationary Distribution:\n", stationary_dist)