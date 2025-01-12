import sympy as sm
import math as m
import matplotlib.pyplot as plt
import numpy as np
import uncertainties as un

## Part 1
#a)
def nArr(n):
    # create array starting at 21 and going to size of nxn
    a = np.arange(0, n*n, 1, dtype=float) + 21
    # reshape into 2d array
    return a.reshape(n, n)


def lower(A):
    # copy the array input
    lowA= np.copy(A)
    n = len(A)
    
    # loop starts with the last two rows, subtracting to get updated rows with zeros starting on the RHS
    for j in range(n-1, -1, -1):
        for i in range(j-1, -1, -1):
            C = lowA[i][j] / lowA[j][j]
            lowA[i][0:j+1] -= C*lowA[j][0:j+1]
    return lowA

def upper(A):
    # copy the array input
    upA= np.copy(A)
    n = len(A)
    
        
    # loop starts with the first two rows, subtracting to get updated rows with zeros starting on the LHS
    for j in range(n-1):
        for i in range(j+1, n):
            # checking if the value is small, adding 1*10^10 to the whole row to avoid rounding errors
            if(abs(upA[j][j]) <= 10e-20):
                upA[j] += 1e10
                
            # checking if the value is zero, performing a row swap with max value in column if so
            if(upA[j][j] == 0):
                k = np.argmax(np.abs(upA[j:, j]))
                upA[[j, n-k-1], :] = upA[[n-k-1, j], :]
                
            C = upA[i][j] / upA[j][j]
            upA[i][j:] -= C*upA[j][j:]
    return upA
    
def eucNorm(A):
    #just square rooting the sum of the whole matrix squared
    return np.sqrt(np.sum(abs(A)**2))

def infNorm(A):
    #returning the maximum of each column sum
    return np.max(np.sum(A, axis=1))

A = nArr(4)
print("Array is:")
print(A)
print("\nUpper Triangular Matrix:")
print(upper(A))
print("\nLower Triangular Matrix:")
print(lower(A))

print("\nEuclidean Norm:")
print(eucNorm(A))
print("\nInfinite Norm:")
print(infNorm(A))

print("\nNumPy Euclidean Norm:")
print(np.linalg.norm(A))
print("\nNumPy Infinite Norm:")
print(np.linalg.norm(A, np.inf))

#b)
def weirdMatrix(n):
    # creating the matrix with 1's on the diagonal, and -1s on the upper triangle
    A = np.ones(n*n) * -1
    A = A.reshape(n, n)
    A[np.diag_indices(n)] = 1 
    return np.triu(A, 0)

def weirdBS(n):
    # creating a 1d array with alternating 1's and -1's
    A = np.empty((n,))
    A[::2] = 1
    A[1::2] = -1
    return A

print("\nMatrix Size of n = 4:")
A = weirdMatrix(4)
BS = weirdBS(4)
print("Condition Number:")
print(eucNorm(np.linalg.inv(A)) * eucNorm(A))
x = np.linalg.solve(A, BS)
print("First 3 x's Without -0.001 Peterbution:")
print(x[0:3])
A[len(A) - 1][0] -= 0.001
x = np.linalg.solve(A, BS)
print("First 3 x's With -0.001 Peterbution:")
print(x[0:3])

print("\nMatrix Size of n = 16:")
A = weirdMatrix(16)
BS = weirdBS(16)
print("Condition Number:")
print(eucNorm(np.linalg.inv(A)) * eucNorm(A))
x = np.linalg.solve(A, BS)
print("First 3 x's Without -0.001 Peterbution:")
print(x[0:3])
A[len(A) - 1][0] -= 0.001
x = np.linalg.solve(A, BS)
print("First 3 x's With -0.001 Peterbution:")
print(x[0:3])   

##Part 2
#a)
def backsub1 (U, bs):
    # UNOPTIMAL SOLUTION PROVIDED FROM LECTURE NOTES
    n = bs.size
    xs = np.zeros(n)
    xs[n-1] = bs[n-1]/U[n-1, n-1] 
    for i in range (n-2, -1, -1):
        bb = 0
        for j in range (i+1, n):
            bb += U[i, j]*xs[j]
        xs[i] = (bs[i] - bb)/U[i, i]
    return xs

def backsub2(U, bs):
    # no BB value
    n = bs.size
    xs = np.zeros(n)
    xs[n-1] = bs[n-1]/U[n-1, n-1] 
    for i in range (n-2, -1, -1):
        ## instead of summing each value individually, nultiply the matrixes and sum
        xs[i] = (bs[i]-np.sum(U[i]*xs))/U[i, i]
    return xs

def mysolve (f, A, bs):
    xs = f(A, bs)
    print ('\nMy solution is:', xs[0], xs[1], xs[2])
    
A = nArr(5000)
U = upper(A)
bs = U[0]

from timeit import default_timer
timer_start = default_timer()
mysolve ( backsub1, U, bs)
timer_end = default_timer()
time1 = timer_end - timer_start
print ('Time 1: ', time1)

from timeit import default_timer
timer_start = default_timer ()
mysolve (backsub2, U, bs)
timer_end = default_timer ()
time2 = timer_end - timer_start
print ('Time 2: ', time2 )

#b) and c)
def gauss(A, bs):
    # get the upper triangular matrix
    upA= np.copy(A)
    upBS = np.copy(bs)
    n = len(A)
    for j in range(n-1):
        for i in range(j+1, n):
            if(abs(upA[j][j]) <= 10e-20):
                upA[j] += 1e10
                bs[j] += 1e10
            if(upA[j][j] == 0):
                k = np.argmax(np.abs(upA[j:, j]))
                upA[[j, n-k-1], :] = upA[[n-k-1, j], :]
                bs[j], bs[n-k-1] = bs[n-k-1], bs[j]
            C = upA[i][j] / upA[j][j]
            upA[i][j:] -= C*upA[j][j:]
            upBS[i] -= C * upBS[j]
            
    # input the upper triangular matrix into backsub2
    return backsub2(upA, upBS)

A = np.array([2.0, 1.0, 1.0, 1.0, 1.0, -2.0, 1.0, 2.0, 1.0])
A = A.reshape(3, 3)
bs = np.array([8.0, -2.0, 2.0])
print("\nGaussian Elimination Matrix 1:", gauss(A, bs))

A = np.array([2.0, 1.0, 1.0, 2.0, 1.0, -4.0, 1.0, 2.0, 1.0])
A = A.reshape(3, 3)
bs = np.array([8.0, -2.0, 2.0])
print("Gaussian Elimination Matrix 2:", gauss(A, bs))