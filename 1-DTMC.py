import numpy as np

State_Space = {'A', 'B', 'C'}

# Probabailty Transition Matrix
# This one is a Stochastic Matrix from Left to Right
P = np.array([
                [0.8, 0.0, 0.2],
                [0.2, 0.7, 0.1],
                [0.3, 0.3, 0.4]
            ])

# To find the next state you square P so:
Pnext = np.matmul(P, P)

# if we have an initial distribution π at time t = 0, say:
π_0 = np.array([0.6, 0.3, 0.1])

# to calculate the next distribution you do:
π_next =  np.matmul(π_0, P)

# this leads to the study of long term behavior where π = π * P
# To find the convergence matrix (stationary distribution):
eigenvalues, eigenvectors = np.linalg.eig(P.T)
stationary = eigenvectors[:, np.isclose(eigenvalues, 1)].real
stationary = stationary / np.sum(stationary)

print("Stationary distribution:", stationary.flatten())

# To verify convergence:
for i in range(100):
    π_0 = np.matmul(π_0, P)

print("Distribution after 100 iterations:", π_0)

# if you want to know the transition probability between states(note here, we are missing X1)
#  P(X_2​=j∣X_0​=i)= ∑(k=0 to n-1) of p(i,k)p(k,j)
# i - initial state; j - final state; k = range from 0 to n

P_example = np.array([
                [0.7, 0.2, 0.1],
                [0.3, 0.5, 0.2],
                [0.2, 0.4, 0.4]
            ])
# to find X1, since we have: P(X2 = 1 | X0 = 2), we do:
# To find P(X2 = 1 | X0 = 2), we do:
i = 2  # Initial state
j = 1  # Final state

# Calculate the probability
prob = sum(P_example[i, k] * P_example[k, j] for k in range(P_example.shape[0]))
print(f"Transition probability P(X2 = {j} | X0 = {i}): {prob}")

'''
The point of using .shape[0]:

Clarity: It explicitly indicates you're working with the first dimension of a potentially multi-dimensional array.
Consistency: If you're using NumPy throughout, sticking to NumPy methods maintains consistency.
Flexibility: .shape allows access to other dimensions easily (e.g., .shape[1] for columns).
'''