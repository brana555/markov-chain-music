import numpy as np

#Markov chain
state = {
    0 : "burger",
    1 : "Pizza",
    2 : "Hotdog"
}
print(state)

#transition matrix
A = np.array([[0.2, 0.6, 0.2], [0.3, 0.0, 0.7], [0.5, 0.0, 0.5]])
print(A)

#Random walk on markov chain
n = 15
start_state = 0
print(state[start_state], "--->", end=" ")
prev_state = start_state

while n-1:
    curr_state = np.random.choice([0, 1, 2], p=A[prev_state])
    print(state[curr_state], "--->", end=" ")
    prev_state = curr_state
    n-=1
print("stop")

#How to find the stationary distribution of our markov chain
#Approach 1: Monte Carlo
steps = 10**6
start_state = 0
pi = np.array([0, 0, 0])
pi[start_state] = 1
prev_state = start_state

i = 0
while i < steps:
    curr_state = np.random.choice([0, 1, 2], p=A[prev_state])
    pi[curr_state]+=1
    prev_state = curr_state
    i +=1 

print("pi = ", pi/steps)


#Approach 2: Repeated Matrix Multiplication
steps = 10**3
A_n = A
i=0
while i<steps:
    A_n = np.matmul(A_n, A)
    i+=1

print("A^n = \n", A_n, "\n")
print("pi = ", A_n[0])

#Approach 3: Finding Left eigen vectors
import scipy.linalg
values, left = scipy.linalg.eig(A, right = False, left = True)

print("left eigen vectors = \n", left, "\n")
print("eigen values = \n", values)

..00
pi = left[:,0]
pi_normalized  = [(x/np.sum(pi)).real for x in pi]
print(pi_normalized)

#P(Pizza --> Hotdog --> Hotdog --> Burger) = ?
#=> P(X_0 = Pizza, X_1 = Hotdog, X_2 = Hotdog, X_3 = Burger)
#=> P(X_0 = Pizza) P(X_1 = Hotdog | X_0 = Pizza) P(X_2 = Hotdog | X_1 = Hotdog) P(X_3 = Burger | X_2 = Hotdog)
def find_prob(seq, A, pi):
    start_state = seq[0]
    prob = pi[start_state]
    prev_state, curr_state = start_state, start_state
    for i in range(1, len(seq)):
        curr_state = seq[i]
        prob *= A[prev_state][curr_state]
        prev_state = curr_state
    return prob

print(find_prob([1, 2, 2, 0], A, pi_normalized))