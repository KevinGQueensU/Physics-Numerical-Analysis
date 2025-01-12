import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
import math as m
import random
from sympy.abc import x


## PART 1 ##
#graph configuration function
def plotSet(title, xLabel, yLabel, xStart, xEnd):
    # font size and res
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.dpi'] = 120

    #figure and axis sizes, labels
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_axes([0.1, 0.1, 0.84, 0.84])
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xlim(xStart, xEnd)

## a)
# function subroutine
def f(x):
    return sm.exp(sm.sin(2 * x))


# derivatives
first = sm.diff(sm.exp(sm.sin(2 * x)), x)
second = sm.diff(first, x)
third = sm.diff(second, x)

# lambda functions
firstLm = sm.lambdify(x, first)
secondLm = sm.lambdify(x, second)
thirdLm = sm.lambdify(x, third)

# define an array with 200 steps
xs = np.linspace(0, np.pi * 2, 200)
print(len(xs), xs[0], xs[len(xs) - 1] - np.pi * 2)

# evaluate at each point
func = [f(x) for x in xs]
fs = [firstLm(x) for x in xs]
fps = [secondLm(x) for x in xs]
fpps = [thirdLm(x) for x in xs]

# graph functions
lw = 2
plotSet(r'$f(x) = e^{sin(2x)}$', r'$x$', r'$f,f^\prime,f^{\prime\prime}, f^{\prime\prime\prime}$', 0, 2 * m.pi)
plt.plot(xs, func, 'y', linewidth=lw)
plt.plot(xs, fs, 'b', linewidth=lw)
plt.plot(xs, fps, 'r', linewidth=lw)
plt.plot(xs, fpps, 'g', linewidth=lw)
plt.legend([r'$f(x)$', r'$f^\prime(x)$', r'$f^{\prime\prime}(x)$', r'$f^{\prime\prime\prime}(x)$'], loc = 'best')
plt.show()


## b)
# forward and central difference subroutines
def fd(x, h):
    return (f(x + h) - f(x)) / h


def cd(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def fdArray(xsa, h):
    fdxs = [fd(x, h) for x in xsa]
    return fdxs


def cdArray(xsa, h):
    cdxs = [cd(x, h) for x in xsa]
    return cdxs

# plot h of 0.15
h = 0.15

# graph configuration
plotSet('Finite Difference: h = 0.15', r'$x$', r'$f^\prime$', 0, 2 *m.pi)
plt.plot(xs, fdArray(xs, h), 'r', linewidth=lw)
plt.plot(xs, cdArray(xs, h), 'y', linewidth=lw)
plt.plot(xs, fs, 'b', linewidth=lw)
plt.legend([r'$fd$', r'$cd$', r'$f^{\prime}(x)$'], loc = 'best')
plt.show()

# plot h of 0.5
h = 0.5

# graph configuration
plotSet('Finite Difference: h = 0.5', r'$x$', r'$f^\prime$', 0, 2 *m.pi)
plt.plot(xs, fdArray(xs, h), 'r', linewidth=lw)
plt.plot(xs, cdArray(xs, h), 'y', linewidth=lw)
plt.plot(xs, fs, 'b', linewidth=lw)
plt.legend([r'$fd$', r'$cd$', r'$f^{\prime}(x)$'], loc = 'best')
plt.show()


## c)
# subroutines for fd and cd error
def fdEps(steps, xVal):
    return [(h * abs(secondLm(xVal))) / 2 + (2 * abs(f(xVal)) * np.finfo(float).eps) / h for h in steps]


def cdEps(steps, xVal):
    return [(((h ** 2) / 24) * abs(thirdLm(xVal)) + (2 * abs(f(xVal)) * np.finfo(float).eps) / h) for h in steps]


# subroutines for absolute error
def fdAbsError(steps, xVal):
    return [abs(firstLm(xVal) - fd(xVal, h)) for h in steps]


def cdAbsError(steps, xVal):
    return [abs(firstLm(xVal) - cd(xVal, h)) for h in steps]


# compute errors
hStep = np.arange(10 ** (-16), 1 + 0.01, 0.01)
fdE = fdEps(hStep, 1)
cdE = cdEps(hStep, 1)
fdAbsE = fdAbsError(hStep, 1)
cdAbsE = cdAbsError(hStep, 1)

# graph configuration
plotSet('Error as a Function of h', r'$h$', r'$|abs. error|$', hStep[0], hStep[len(hStep) - 1])
plt.plot(hStep, fdE, 'r', linewidth=lw)
plt.plot(hStep, cdE, 'y', linewidth=lw)
plt.plot(hStep, fdAbsE, 'b', linewidth=lw)
plt.plot(hStep, cdAbsE, 'g', linewidth=lw)
plt.yscale('log')
plt.xscale('log')
plt.legend([r'$\varepsilon_{fd}$', r'$\varepsilon_{cd}$', r'$f^\prime - fd$', r'$f^\prime - cd$'], loc = 'best')
plt.show()


## PART 2 ##
## a)
# function in Rodrigues
def f(x, n):
    return (x ** 2 - 1) ** n

# non-recursive central difference for n = 1 to 4
def cd1(x, h, n):
    return (f(x + h, n) - f(x - h, n)) / (2 * h)

def cd2(x, h, n):
    return (cd1(x + h, h, n) - cd1(x - h, h, n)) / (2 * h)

def cd3(x, h, n):
    return (cd2(x + h, h, n) - cd2(x - h, h, n)) / (2 * h)

def cd4(x, h, n):
    return (cd3(x + h, h, n) - cd3(x - h, h, n)) / (2 * h)

def cd(x, n, h):
    if n == 1:
        return cd1(x, h, 1)
    if n == 2:
        return cd2(x, h, 2)
    if n == 3:
        return cd3(x, h, 3)
    if n == 4:
        return cd4(x, h, 4)

# Rodrigues subroutine
def rodF(n, xsa, h):
    return [1 / (2 ** n * m.factorial(n)) * cd(x, n, h) for x in xsa]

# settings for computing
x1 = -1
x2 = 1
h = 0.01

# creating array
xs = np.arange(x1, x2 + h, h)

# graph configuration
plotSet('Legendre Polynomials', r'$x$', r'$f(x)$', x1, x2)
plt.plot(xs, rodF(1, xs, h), 'r', linewidth=lw)
plt.plot(xs, rodF(2, xs, h), 'y', linewidth=lw)
plt.plot(xs, rodF(3, xs, h), 'b', linewidth=lw)
plt.plot(xs, rodF(4, xs, h), 'g', linewidth=lw)
plt.legend([r'$n = 1$', r'$n = 2$', r'$n = 3$', r'$n = 4$'], loc='best')
plt.show()

## b)
# recursive subroutine for central difference
def cdRec(n, x, n1, h):  # n1 is a variable for keeping count of # of recursive calls, initially equal to n
    if n1 == 1:
        return (f(x + h, n) - f(x - h, n)) / (2 * h)  # base case
    return (cdRec(n, x + h, n1 - 1, h) - cdRec(n, x - h, n1 - 1, h)) / (2 * h)  # recursive calling

# Rodrigues subroutines
def rodFRec(n, xsa, h):
    return [1 / (2 ** n * m.factorial(n)) * cdRec(n, x, n, h) for x in xsa]

def rodPlt(n, xsa, h):
    r = lambda: random.randint(0, 255)  # generating random color
    legend = []  # storing legend information
    for i in range(1, n + 1):
        color = '#%02X%02X%02X' % (r(), r(), r())  # converting color to hex code
        legend.append('n = ' + str(i))
        plt.plot(xsa, rodFRec(i, xsa, h), color, linewidth=2)  # plot the Rodrigues value for value n
    plt.legend(legend, loc='best')  # update legend

# graph configuration
plotSet('Legendre Polynomials', r'$x$', r'$f(x)$', x1, x2)
rodPlt(8, xs, h)
plt.show()
