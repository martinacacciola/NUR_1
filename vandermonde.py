#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import timeit

'''
This script contains the code for Exercise 2.
'''

'''
Point a):

We implement the LU decomposition using Crout's algorithm to solve the Vandermonde system.
The problem is divided into:

1. LU decomposition: the matrix A is decomposed into a lower triangular matrix L and an upper triangular matrix U.
Firstly we initialize L and U as zero matrices of the same size as A. 
Setting alpha_ii is done implicitly in passage 1: 
since L[i, :i] and U[:i, i] are both zero vectors at this point this dot product is zero, and so U[i, i] is just A[i, i].
Then we update the elements of L and U loop over the columns.


2. Forward substitution: we solve Ly = b for y. 
We initialize y as a zero vector of the same size as b. 
Then we solve the system iteratively for each element of y starting from first element.

3. Backward substitution: we solve Ux = y for x.
We initialize x as a zero vector of the same size as y.
Then we solve the system iteratively for each element of x starting from the last element
'''

# LU Decomposition 
# beta_ij corresponds to U[i, j], and alpha_ij corresponds to L[i, j]

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Loop over the columns
    for i in range(n):
        # Update the upper and lower triangular matrices
        U[i, i:] = A[i, i:] - np.dot(L[i, :i], U[:i, i:]) # Passage 1
        L[i:, i] = (A[i:, i] - np.dot(L[i:, :i], U[:i, i])) / U[i, i] # Passage 2

    return L, U

# Forward substitution
def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double)
    # Solve Ly = b for y
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i] # Passage 3
    return y

# Backward substitution
def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)
    # Solve Ux = y for x
    x[-1] = y[-1] / U[-1, -1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i] # Passage 4
    return x

# Load the data
data = np.genfromtxt("./Vandermonde.txt", comments='#', dtype=np.float64)
x = data[:, 0]
y = data[:, 1]

# Create the Vandermonde matrix
def create_vandermonde(x, N=None):
    if N is None:  # If N is not provided, use the length of x
        N = len(x)
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            V[i, j] = x[i] ** j
    return V

V = create_vandermonde(x)

# LU decomposition using Crout's algorithm
L, U = lu_decomposition(V)

# Solve the system
y_forward = forward_substitution(L, y)
c = backward_substitution(U, y_forward)

# Save the coefficients to a text file
np.savetxt('coefficients.txt', c)

# Plot the 19-th degree polynomial evaluated at 1000 equally-spaced points
# Along with the original data points
# This plot is just to have another look at the region of interest (it is the same as the next one)
x_values = np.linspace(0, 100, 1000)
y_values = np.zeros_like(x_values)
for i in range(len(c)):
    y_values += c[i] * x_values ** i
plt.plot(x_values, y_values, label="19-th degree polynomial")
plt.ylim(-1000, 6000)
plt.scatter(x, y, color='red', label='Data points')
plt.legend()
plt.savefig('plots/polynomial_plot.png')


xx=np.linspace(x[0],x[-1],1001) #x values to interpolate at
# Calculate interpolated y values
yya = np.array([np.sum(c * xi ** np.arange(len(c))) for xi in xx])
ya = np.array([np.sum(c * xi ** np.arange(len(c))) for xi in x])

# Plot 2a
fig=plt.figure()
gs=fig.add_gridspec(2,hspace=0,height_ratios=[2.0,1.0])
axs=gs.subplots(sharex=True,sharey=False)
axs[0].plot(x,y,marker='o',linewidth=0)
plt.xlim(-1,101)
axs[0].set_ylim(-400,400)
axs[0].set_ylabel('$y$')
#axs[1].set_ylim(1e-16,1e1)
axs[1].set_ylim(1e-20,1e1)
axs[1].set_ylabel('$|y-y_i|$')
axs[1].set_xlabel('$x$')
axs[1].set_yscale('log')
line,=axs[0].plot(xx,yya,color='orange')
line.set_label('Via LU decomposition')
axs[0].legend(frameon=False,loc="lower left")
axs[1].plot(x,abs(y-ya),color='orange')
plt.savefig('plots/my_vandermonde_sol_2a.png',dpi=300)


'''
Point b):
We implement Neville's algorithm to interpolate the data points.
1) We start by identifying the M tabulated points closest to the point x0
To do so, we sort the points by their distance to x0 (using the function argsort) and select the M closest points.
2) We set the initial p_i = y_i for i in 0 to M.
3) We loop over each order of the interpolation from 1 to M-1
For each order, we loop over each interval of the current order from 0 to M-k-1
and update the p value for the interval, overwriting the previous orders' values.
4) We return the first element of p, which is the interpolated value.


'''

def argsort(seq):
    # Pair each element in the sequence with its index
    indexed_seq = [(val, i) for i, val in enumerate(seq)]
    # Sort the indexed sequence
    sorted_indexed_seq = sorted(indexed_seq)
    # Extract the indices from the sorted indexed sequence
    sorted_indices = [i for val, i in sorted_indexed_seq]
    return sorted_indices


def neville(x, y, x0, M):
    '''
    Parameters:
    x: x-coordinates of the data points
    y: y-coordinates of the data points
    x0: x-coordinate to interpolate at
    M: number of points to use for interpolation
    Returns:
    p[0]: interpolated value at x0
    '''
    # Sort the points by their distance to x0 and select the M closest points
    sorted_indices = argsort(np.abs(x - x0))
    x = x[sorted_indices[:M]]
    y = y[sorted_indices[:M]]

    # Set p = y
    p = np.copy(y) 

    # Loop over each order of the interpolation
    for k in range(1, M):
        # Loop over each interval of the current order
        for i in range(M-k):
            # Update the p value for the interval
            p[i] = ((x0 - x[i+k])*p[i] + (x[i] - x0)*p[i+1]) / (x[i] - x[i+k])
    return p[0] # Return the first element which is the interpolated value


# Values for interpolation
xx = np.linspace(0, 100, 1000)

# Interpolation using Neville's algorithm
yyb = np.array([neville(x, y, xi) for xi in xx])
yb = np.array([neville(x, y, xi) for xi in x])

# Plotting
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
axs = gs.subplots(sharex=True, sharey=False)
axs[0].plot(x, y, marker='o', linewidth=0)
plt.xlim(-1, 101)
axs[0].set_ylim(-400, 400)
axs[0].set_ylabel('$y$')
axs[1].set_ylim(1e-16, 1e1)
axs[1].set_ylabel('$|y-y_i|$')
axs[1].set_xlabel('$x$')
axs[1].set_yscale('log')
line, = axs[0].plot(xx, yyb, linestyle='dashed', color='green')
line.set_label("Via Neville's algorithm")
axs[0].legend(frameon=False, loc="lower left")
axs[1].plot(x, abs(y - yb), linestyle='dashed', color='green')
plt.savefig('plots/my_vandermonde_sol_2b.png', dpi=300)

'''
Point c):
Iterative version of LU decomposition.
To do so, we start by saving the value of the initial guess x0
(this passage is necessary to avoid overwriting the initial guess during iterations).

We then loop over the number of iterations and perform the following steps:
1) Calculate delta_b from Ax'=b+delta_b where x'=x0+delta_x (the imperfect solution)
2) Perform LU decomposition
3) Solve Ly = δb for y
4) Solve U δx = y for δx. This δx is used to improve the solution x0: x''=x'-δx (where x' is the current imperfect solution)

'''
# Function for iterative solution using LU decomposition
def iterative_improvement(A, b, x0, iterations):
    '''
    Inputs:
    A: matrix 
    b: vector
    x0: initial guess for the solution
    iterations: number of iterations to perform
    Returns:
    x: improved solution
    '''
    
    x = np.copy(x0)
    for _ in range(iterations):
        # Difference between actual b and b calculated using current solution x
        delta_b = np.dot(A, x) - b
        # Perform LU decomposition
        L, U = lu_decomposition(A)
        # Solve Ly = δb for y
        y = forward_substitution(L, delta_b)
        # Solve U δx = y for δx
        delta_x = backward_substitution(U, y)
        # Improve the solution
        x -= delta_x
    return x


# Perform 1 LU iteration
c1 = iterative_improvement(V, y, c, 1)
yya1 = np.array([np.sum(c1 * xi ** np.arange(len(c1))) for xi in xx])
ya1 = np.array([np.sum(c1 * xi ** np.arange(len(c1))) for xi in x])

# Perform 10 LU iterations
c10 = iterative_improvement(V, y, c, 10)
yya10 = np.array([np.sum(c10 * xi ** np.arange(len(c10))) for xi in xx])
ya10 = np.array([np.sum(c10 * xi ** np.arange(len(c10))) for xi in x])

# Plot 2c
fig=plt.figure()
gs=fig.add_gridspec(2,hspace=0,height_ratios=[2.0,1.0])
axs=gs.subplots(sharex=True,sharey=False)
axs[0].plot(x,y,marker='o',linewidth=0)
plt.xlim(-1,101)
axs[0].set_ylim(-400,400)
axs[0].set_ylabel('$y$')
axs[1].set_ylim(1e-16,1e3)
axs[1].set_ylabel('$|y-y_i|$')
axs[1].set_xlabel('$x$')
axs[1].set_yscale('log')
line,=axs[0].plot(xx,yya1,linestyle='dotted',color='red')
line.set_label('LU with 1 iteration')
axs[1].plot(x,abs(y-ya1),linestyle='dotted',color='red')

line,=axs[0].plot(xx,yya10,linestyle='dashdot',color='purple')
line.set_label('LU with 10 iterations')
axs[1].plot(x,abs(y-ya10),linestyle='dashdot',color='purple')

axs[0].legend(frameon=False,loc="lower left")
plt.savefig('plots/my_vandermonde_sol_2c.png',dpi=300)

'''
Point d): Exection time comparison

To measure the execution time of the three methods, we use the timeit module.
We run each method 100 times (specified by 'numnber') and save the results to a text file.

 - globals() is used to pass the variables defined in the code to the timeit function
 - the first instance of timeit is used to measure the execution time
 - the second instance of timeit is used to display the results obtained in previous step
 
'''

# (a) LU decomposition
lu_time = timeit.timeit("lu_decomposition(V)", globals=globals(), number=100)

# (b) Neville's algorithm
neville_time = timeit.timeit("np.array([neville(x, y, xi, 20) for xi in xx])", globals=globals(), number=100)

# (c) Iterative improvement with LU decomposition (10 iterations)
iterative_time = timeit.timeit("iterative_improvement(V, y, c, 10)", globals=globals(), number=100)

# Write results to a text file
with open('timing_results.txt', 'w') as file:
    file.write(f"LU Decomposition Time: {lu_time:.6f} seconds\n")
    file.write(f"Neville's Algorithm Time: {neville_time:.6f} seconds\n")
    file.write(f"Iterative Improvement Time: {iterative_time:.6f} seconds\n")




