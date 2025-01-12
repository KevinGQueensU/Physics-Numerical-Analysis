import sympy as sm
import numpy as np
import math as m
import mpmath as mp
from scipy.integrate import dblquad


## Part 1
# a)
def f(x):
    return np.exp(-x ** (2))

def erfRect(f, a, b, n):
    h = (b - a) / (n - 1)
    xi = a + np.arange(n - 1) * h
    return np.sum((2 / (np.pi ** 0.5)) * f(xi) * h)

# b)
def relErr(ansNum, x):
    return abs((ansNum - m.erf(x))/ansNum)

def erfTrap(f, a, b, n):
    h = (b - a) / (n - 1)
    cs = np.array([(1/2)])
    cs = np.append(cs, np.tile(1, n - 2))
    cs = np.append(cs, 0.5)
    xi = a + np.arange(n) * h
    return np.sum((2 / (np.pi ** 0.5)) * f(xi) * cs * h)

def simpArr(n):
    c = np.array([1])
    c1 = np.tile([4, 2], int((n - 2) / 2))
    c2 = np.append(c, c1)
    cs = np.append(c2, [4, 1])
    return cs

def erfSimp(f, a, b, n):
    if(n % 2 == 0):
        n += 1
        print("ERROR: Even number of points detected, incrementing.")
    h = (b - a) / (n - 1)
    cs = simpArr(n)
    xi = a + np.arange(n) * h
    return np.sum(f(xi) * cs * h/3 * (2 / (np.pi ** 0.5)))

def adaptive_step(erf, a, b, err):
    n = 3
    while relErr(erf(f, a, b, n), b) > abs(err):
        n = 2 * n - 1
    return n

intRect = erfRect(f, 0, 1, 100)
print("Rectangle Rule: " + str(intRect))
print("Relative Error: " + str(100 * relErr(intRect, 1)) + "% \n")

intTrap = erfTrap(f, 0, 1, 100)
print("Trapezoidal Rule 100 Steps: " + str(intTrap))
print("Relative Error: " + str(100 * relErr(intTrap, 1)) + "%")
print("Steps required for 10e-10 error: " + str(adaptive_step(erfTrap, 0, 1, 10**(-10))))

intTrap = erfTrap(f, 0, 1, 101)
print("\nTrapezoidal Rule 101 Steps: " + str(intTrap))
print("Relative Error: " + str(100 * relErr(intTrap, 1)) + "%")
print("Steps required for 10e-10 error: " + str(adaptive_step(erfTrap, 0, 1, 10**(-10))))

print("\nFor trapezoidal rule, there is no rule on if the number of points have to be even or odd. Above you can see "
      "that 100 and 101"
      "\nsample points both compute fine, and 101 is more accurate. This is because it's just creating trapezoids "
      "between two points.")
print("\nMy implementation of Simpson's rule breaks when even steps are forcefully inputted, so I made it increment "
      "if it detected an even number."
      "\nIt breaks because the version is using the traditional 1/3 panelling, meaning that it REQUIRES an "
      "odd number of points"
      " to make an even number of panels. \nBelow, I input 100 and it increments my value, then displays a message. \n")

intSimp = erfSimp(f, 0, 1, 100)
print("Simpson Rule 100 Steps: " + str(intSimp))
print("Relative Error: " + str(100 *relErr(intSimp, 1)) + "%")
print("Steps required for 10e-10 error: " + str(adaptive_step(erfSimp, 0, 1, 10**(-10))))

intSimp = erfSimp(f, 0, 1, 101)
print("\nSimpson Rule 101 Steps: " + str(intSimp))
print("Relative Error: " + str(100 *relErr(intSimp, 1)) + "%")
print("Steps required for 10e-10 error: " + str(adaptive_step(erfSimp, 0, 1, 10**(-10))))


## Part 2
# a)
def f2d(x, y):
    return (x**2 + y)**0.5 * np.sin(x) * np.cos(y)

def simp2d(f,a,b,c,d,n,m):
    if(n % 2 == 0):
        print("ERROR: Even number of y points detected, incrementing.")
        n += 1
    if (m % 2 == 0):
        print("ERROR: Even number of x points detected, incrementing.")
        m += 1

    hx = (b - a) / (n - 1)
    hy = (d - c) / (m - 1)

    cx = simpArr(n) * hx/3
    cy = simpArr(m) * hy/3
    cw = np.outer(cx, cy)

    xi = a + np.arange(n) * hx
    yi = c + np.arange(m) * hy
    x, y = np.meshgrid(xi, yi, indexing='ij')

    return(np.sum(cw * f(x,y)))

print("\nSimpsons (n, m = 101, 101): " + str(simp2d(f2d, 0, np.pi, 0, np.pi/2, 101, 101)))
print("Simpsons (n, m = 51, 101): " + str(simp2d(f2d, 0, np.pi, 0, np.pi/2, 51, 101)))
print("Simpsons (n, m = 1001, 1001): " + str(simp2d(f2d, 0, np.pi, 0, np.pi/2, 1001, 1001)))

# b)
def quad(a, b, c, d):
    fL = lambda x,y: (x**2 + y)**0.5 * sm.sin(x) * sm.cos(y)
    return mp.quad(fL, [a, b], [c, d])

print("Quad : " + str(quad(0, np.pi, 0, np.pi/2)))

# c)
print("Double Quad: " + str(dblquad(f2d, 0, np.pi/2, 0, np.pi))) #weird notation thing, it inputs y first then x so I had to swap the bounds