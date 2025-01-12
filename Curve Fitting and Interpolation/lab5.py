#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:56:11 2024

@author: kevingao
"""
import sympy as sm
import math as m
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import scipy as sci

def plotSet(title, xLabel, yLabel, xStart, xEnd, yStart, yEnd):
    # font size and res
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.dpi'] = 240

    #figure and axis sizes, labels
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_axes([0.1, 0.1, 2, 1])
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xlim(xStart, xEnd)
    ax.set_ylim(yStart, yEnd)
    
#%% Part 1

def rung(x):
    return 1/(1+25*x**2)

# Function that generates either Chebyshev nodes or equidistant nodes
def genNode(n, f, opt):
    if(opt == "cheb"): 
        j = np.arange(0, n, 1) # This is the j index used for creating the Chebyshev nodes
        xs = -np.cos((j*np.pi)/(n-1)) # Creating the nodes
        return xs, f(xs)
    if(opt == "eq"):
        xs = np.linspace(-1, 1, n) # Creating equidistant notes
        return xs, f(xs)

# Function that outputs the kth Lagrange Polynomial, x is an input, xn is a node,
def lk(x, xn, k):
    lks = np.empty(len(x)) # Empty array for storing each result
    xj = np.concatenate((xn[:k], xn[k+1:])) # Basically a Kronecker Delta for j != k
    for i in range(len(x)):
        lks[i] = np.prod(x[i]-xj)/np.prod(xn[k]-xj) # Taking the products for each input x point
    return lks

# Function that outputs the y values for xn and yn nodes, for x amount of points using Lagrange Basis
def lagrange(x, xn, yn):
    ys = 0
    for k in range(len(xn)):
        ys += (yn[k]*lk(x, xn, k)) # Computing polynomial for x points
        
    # I originally wanted to store the Lagrange polynomials as lambda functions in an array but I could not figure out
    # how to store lambda functions in a Numpy array so I had to use iteration instead of matrix multiplication.
    
    return ys

# Function that outputs the y values for xn and yn nodes, for x amount of points using Monomial Basis
def mon(x, xn, yn):
    n = len(xn)
    m = np.ones([n, n]) # Creating an n x n ones matrix
    
    for i in range(1, n):
        m[:, i] = xn**i # Creating the Vandermonde matrix column by column
        
    cs = np.linalg.solve(m, yn) # Solving for the coeffecients of Monomial
    ys = np.tile(cs[0], len(x)) # First adding the 0th coeffecient to all y's
    
    for i in range(1, n):
        ys += x**i * cs[i] # Adding the rest of the coeffecients multiplied by x to the i

    return ys


xs, ys = genNode(15, rung, "eq")
x = np.linspace(-1, 1, 100)
plotSet("Equidistant Nodes", "x", "f(x)", -1.1, 1.1, -1, 7.6)
plt.plot(xs, ys, 'bo', markersize = 10)
plt.plot(x, mon(x, xs, ys), 'g', lw=5)
plt.plot(x, lagrange(x, xs, ys), 'r--', lw=5, dashes=(4, 5))
plt.legend(["Nodes", "Monomial Basis", "Lagragian Basis"])
plt.show()

xs, ys = genNode(15, rung, "cheb")
x = np.linspace(-1, 1, 100)
plotSet("Chebyshev Nodes", "x", "f(x)", -1.1, 1.1, 0, 1.1)
plt.plot(xs, ys, 'bo', markersize = 15)
plt.plot(x, mon(x, xs, ys), 'g', lw=5)
plt.plot(x, lagrange(x, xs, ys), 'r--', lw=5, dashes=(4, 5))
plt.legend(["Nodes", "Monomial Basis", "Lagragian Basis"])
plt.show()

#%% Part 2

# Function that gets the coeffecients for the cubic interpolation
def coeff(xn, yn):
    n = len(xn) - 1 
    
    # Array of the b values for matrix from the xn and yn nodes
    bs = 6*((yn[2:n+1]-yn[1:n])/(xn[2:n+1]-xn[1:n])-(yn[1:n]-yn[0:n-1])/(xn[1:n]-xn[0:n-1])) 
    
    # Creating the diagonal values from the xn nodes
    xs1 = xn[:(n-1)]
    xs2 = xn[2:(n+1)]
    m1 = np.diag(2*(xs2-xs1), 0)
    m2 = np.diag(xs2[:len(xs2)-1]-xs1[1:], 1)
    m3 = np.diag(xs2[:len(xs2)-1]-xs1[1:], -1)
    m = m1 + m2 + m3
    
    return np.linalg.solve(m, bs)

# Function that finds the cubic interpolation for xn and yn nodes, for xsa amount of points
def cubic(xsa, xn, yn):
    ys = np.empty(0)
    cs = np.zeros(xn.size) # Zeros at the start and beginning of coeffecient array
    cs[1:-1] = coeff(xn, yn)

    # Loop that calculates the y values between the k and k-1 node
    for k in range(1, len(xn)):
        
        # Finding the values of xsa where it is in between two nodes
        minmax = np.array(np.where((xsa <= xn[k]) & (xsa >= xn[k-1])))
        minmax = minmax.reshape(-1)
        xs = xsa[minmax[0]:(minmax[-1] + 1)]
        
        # Calculating that big formula all at once, I don't think this is gonna be readible regardless if I separate it or not so I just left it as one big thing
        ys = np.append(ys, yn[k-1]*(xn[k]-xs)/(xn[k]-xn[k-1]) + yn[k]*(xs-xn[k-1])/(xn[k]-xn[k-1]) 
        - (cs[k-1]/6)*((xn[k]-xs)*(xn[k]-xn[k-1]) - (xn[k]-xs)**3/(xn[k]-xn[k-1]))
        - (cs[k]/6)*((xs-xn[k-1])*(xn[k]-xn[k-1]) - (xs-xn[k-1])**3/(xn[k]-xn[k-1])))
        
    return ys

xn = np.array([2.5 , 2.8 , 3.0 , 3.3 , 3.7 , 4.1 , 4.3 , 4.7 ,
5. , 5.2 , 5.6 , 6.1 , 6.5 , 6.7 , 7.1 , 7.2 , 7.5])

yn = np.array([1.084e-17, 6.7041e-11, 1.125e-07, 2.359e-04,
5.749e-02, 5.188e-01, 7.865e-01, 9.919e-01,
1.000e+00, 9.984e-01, 8.784e-01, 2.312e-01,
6.329e-03, 2.359e-04, 3.579e-09, 6.704e-11, 1.084e-17])

itot = 500 
xmin = 2.5
xmax = 7.5

xs = np.zeros(itot)
xs = xmin+(xmax-xmin)*np.arange(itot)/(itot-1)

plotSet("Lagrange Basis", "x", "f(x)", 2, 8, -1.5, 1.2)
plt.plot(xn, yn, 'bo', markersize = 15)
plt.plot(xs, lagrange(xs, xn, yn), 'r', lw = 5)
plt.legend(["Nodes", "Lagrage Basis Interp."])
plt.show()

ys = cubic(xs, xn, yn)
plotSet("Cubic Spline", "x", "f(x)", 2.1, 7.99, -0.199, 1.2)
plt.plot(xn, yn, 'bo', markersize = 15)
plt.plot(xs, ys, 'r', lw = 5)
plt.legend(["Nodes", "Cubic Spline Interp."])

#%% Part 3

def f(x):
    return np.exp(np.sin(2*x))

# Function to generate the nodes for trignometric interpolation
def genNode(n, f):
    j = np.arange(0, n, 1)
    xs = 2*np.pi*j/n
    ys = f(xs)
    return xs, ys

# Function to find the ak values for trig
def ak(xn, yn):
    n = len(xn)
    m = n//2 
    k = np.arange(0, m+1, 1) # the k range for ak, 0 - m
    a = np.zeros(m+1) 
    
    for i in range(m+1):
        a[i] = (1/m) * np.sum(yn[:n] * np.cos(k[i]*xn[:n])) # Finding the values for the ith coeffecient
    return a

# Function to find the bk values for trig
def bk(xn, yn):
    n = len(xn)
    m = n//2
    k = np.arange(0, m, 1) # the k range for bk, should start at 1 but got confused so I just did 0 - m
    b = np.zeros(m) # Same thing here, the b[0] is just 0 cause it doesn't exist
    
    for i in range(1, m):
        b[i] = (1/m) * np.sum(yn[:n] * np.sin(k[i]*xn[:n]))  # Finding the values for the ith coeffecient
    return b

# Function to trigonometrically interpolate data with xn and yn nodes, for xs amount of points
def trig(xs, xn, yn):
    
    # Preliminary stuff, coeffecients, k ranges, ys array
    n = len(xn)
    m = n//2
    a = ak(xn, yn)
    b = bk(xn, yn)
    k = np.arange(0, m+1, 1)
    ys = np.empty(0)
    
    for i in range(len(xs)):
        # Computing that big formula for the ith x point
        ys = np.append(ys, 0.5*a[0] + np.sum(a[1:m] * np.cos(k[1:m] * xs[i]) + b[1:m] * np.sin(k[1:m] * xs[i])) + 0.5*a[-1]*np.cos(k[-1]) * xs[i])
    
    return ys

xn, yn = genNode(11, f)
xs = np.linspace(0, 2*np.pi, 500)
plotSet("Trig. Interpolation: n = 11", "x", "f(x)", -0.1, 2*np.pi*1.05, 0, 3)
plt.plot(xn, yn, 'bo', markersize = 15)
plt.plot(xs, trig(xs, xn, yn), 'r', markersize = 15)
plt.legend(["Nodes", "Trig. Interpolation"])
plt.show()

plotSet("Trig. Interpolation: n = 51", "x", "f(x)", -0.1, 2*np.pi*1.05, 0, 3)
xn, yn = genNode(51, f)
plt.plot(xn, yn, 'bo', markersize = 15)
plt.plot(xs, trig(xs, xn, yn), 'r', markersize = 15)
plt.legend(["Nodes", "Trig. Interpolation"])

#%% BONUS MARKS
# I saw that the distribution was offset by x = 5, so I had a standard gaussian function offset by x = 5
# Then I looked up what super gaussian meant, so I just squared this function in the exponential and got the exact points

def superGauss(x):
    return np.exp(-((x-5)**2)**2)

xs = np.linspace(2.5, 7.5, 1000)
plotSet("Super Gaussian Function", "x", "f(x)", 2.1, 7.99, -0.199, 1.2)
plt.plot(xs, superGauss(xs), 'r')
plt.legend([r"$f(x) = e^{(-{[{(x-5)}^2]}^2)}$"])
