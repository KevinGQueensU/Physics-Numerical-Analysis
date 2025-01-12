import sympy as sm
import math as m
import matplotlib.pyplot as plt
import numpy as np
import uncertainties as un
import scipy as sci
import matplotlib.cm as cm
import matplotlib.colors as col
import numpy.linalg as la
from mpl_toolkits.mplot3d import axes3d
from sympy.abc import x, y

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
    
    
## PART 1
# a)
def f1(x):
    return 1/(x-3)

def bisect(f, x1, x2, tol):
    N = 0
    while (abs(x2-x1))/(2**N) >= tol:
        N += 1
        x3 = (x1+x2)/2
        if(f(x3)*f(x1) < 0):
            x2 = x3
        else:
            x1 = x3
    return x3
        

L = bisect(f1, 0, 5, 10**-9)
xs1 = np.linspace(-5, 2.999, 1000)
xs2 = np.linspace(3.001, 5, 1000)
plotSet(r"$f(x) = \frac{1}{x-3}$", "x", "f(x)", 2.5, 3.5, -100, 100)
plt.plot(xs1, f1(xs1), 'b')
plt.plot(xs2, f1(xs2), 'b')
plt.axvline(3, color = 'r', linestyle = '--')
plt.legend(['f(x)', '_', 'Asymptote @ x = 3'])
print("Bisecection Root Approximation (Part 1 a): x = ", L, '\n')
plt.show()

# b) and c)
def f2(x):
    return np.exp(x-np.sqrt(x))-x
    
xs = np.linspace(0, 5, 1000)
f2lm = sm.exp(x-x**0.5)-x
L2 = bisect(f2, 0, 1, 10**-8)
plotSet(r"$f(x) = exp(x-\sqrt{x})-x$", "x", "f(x)", 0, 3, -0.5, 1)
plt.axhline(0, color = 'k')
plt.plot(xs, f2(xs), 'b')
print("Bisection Root Approximation (Part 1 b): x1 = ", L2)

def newt(fx, xo, tol):
    # Settings for first root
    x2 = xo
    fp = sm.lambdify(x, sm.diff(fx, x))
    f = sm.lambdify(x, fx)
    n = 0
    
    # Loop for finding first root
    while (abs((f(xo))/(fp(xo))) > tol):
        xo = xo - (f(xo))/(fp(xo))
        n += 1
        if(abs((f(xo))/(fp(xo))) > 10**-3):
            plt.text(xo, -0.2, r'$x_{{{:2d}}}$'.format(n) + '=' + "{:.2f}".format(xo), ha='center', color='r', size='large')
        plt.plot(xo, 0, 'ro', markersize = 10)
        xs = np.linspace(xo, (fp(xo)*xo-f(xo))/fp(xo),  1000)
        plt.vlines(xo, ymin = 0, ymax=f(xo), color = 'k', linestyle = '-.')
        plt.plot(xs, fp(xo) * xs + f(xo) - fp(xo)*xo, 'r--')

    # Settings to supress first root
    a = xo
    u = sm.lambdify(x, fx/(x-a))
    up = sm.lambdify(x, sm.diff(fx/(x-a), x))
    
    # Finding second root with suppressed function
    while (abs((u(x2))/(up(x2))) > tol):
        x2 = x2 - (u(x2))/(up(x2))
    plt.plot(x2, 0, 'ro', markersize=10)
    plt.text(x2, -0.2, r'$x_{root2}$' + '=' + "{:.2f}".format(x2), ha='center', color='r', size='large')
    
    return xo, x2

r1, r2 = newt(f2lm, 0.01, 10**-8)
print("Newton's Method Roots:", "x1 = {:2f},".format(r1), "x2 = {:2f}".format(r2), '\n')
plt.show()

## PART 2
def farchi(h, p):
    return 3*h**2-h**3-4*p

def bisectArchi(f, h1, h2, p, tol):
    N = 0
    while (abs(h2-h1))/(2**N) >= tol:
        N += 1
        h3 = (h1+h2)/2
        if(f(h3, p)*f(h1, p) < 0):
            h2 = h3
        else:
            h1 = h3
    return h3

def archis(f, pSph, h1, h2, tol):
    hs = []
    for p in pSph:
        hs.append(bisectArchi(f, h1, h2, p, tol))
    return hs

def archi(f, pSph, h1, h2, tol):
    return bisectArchi(f, h1, h2, pSph, tol)

ps = np.linspace(0, 1, 1000)
ps = ps[1:len(ps) - 1]
plotSet(r"Archimedes Principle: Displacement as a Function of Density (Sphere)", r"$\rho_{sph}$", r"$h$", 0, 1, 0, 2)
plt.plot(ps, archis(farchi, ps, 0, 2, 10**-8))
plt.legend([r'f$(\rho_{sph}) = h$'])
h = archi(farchi, 0.55, 0, 2, 10**-8)

X = np.linspace(-1, 1, 1000)
Y = np.linspace(-1, 1, 1000)
X, Y = np.meshgrid(X, Y)
Z = 0 * (X+Y)

u, v = np.mgrid[0:2*np.pi:500j, 0:np.pi:500j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v) - (1-h)

zColor = np.copy(z)
for i in range(zColor.shape[0]):
    for j in range(zColor.shape[1]):
        if zColor[i][j] <= 0:
            zColor[i][j] = 10
        else:
            zColor[i][j] = 0


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(np.min(z), np.max(z))
ax.set_xlabel('x', fontsize = 20)
ax.set_ylabel('y', fontsize = 20)
ax.set_zlabel('z', fontsize=  20)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.view_init(-160, 60)
plt.title(r"$\rho_{sph}$ = 0.55")
ax.plot_surface(x, y, z, facecolors = cm.tab20b(zColor))
ax.plot_surface(X, Y, Z, color = 'gray', alpha=0.3)
plt.show()

h = archi(farchi, 0.82, 0, 2, 10**-8)
z = np.cos(v) - (1-h)
zColor = np.copy(z)
for i in range(zColor.shape[0]):
    for j in range(zColor.shape[1]):
        if zColor[i][j] <= 0:
            zColor[i][j] = 1
        else:
            zColor[i][j] = 0

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(np.min(z), np.max(z))
ax.set_xlabel('x', fontsize = 20)
ax.set_ylabel('y', fontsize = 20)
ax.set_zlabel('z', fontsize=  20)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.view_init(-160, 30)
plt.title(r"$\rho_{sph}$ = 0.8", x = 0.5, y = 1)
ax.plot_surface(x, y, z, facecolors = cm.tab20b(zColor))
ax.plot_surface(X, Y, Z, color = 'gray', alpha=0.3)

## PART 3
# a) and b)
import numpy as np
import numpy.linalg as la

def fs1(xs):
    x0, x1 = xs[0], xs[1]
    f1 = x0**2 - 2*x0 + x1**4 - 2*(x1**2) + x1
    f2 = x0**2 + x0 + 2*(x1**3) - 2*(x1**2) -1.5*x1 -0.05
    return np.array([f1, f2])

def fs2(xs):
    x0, x1, x2 = xs[0], xs[1], xs[2]
    f1 = 2*x0 - x1 * np.cos(x2) - 3
    f2 = x0**2 - 25*(x1-2)**2 + np.sin(x2) - (np.pi/10)
    f3 = 7 * x0 * np.exp(x1) - 17 * x2 + 8*np.pi
    return np.array([f1, f2, f3])

def jac(fs, xs):
    h = 10**-4
    n = len(xs)
    j = np.ones([n, n])
    for i in range(n):
        xsa = np.copy(xs)
        xsa[i] += h
        j[:, i] = (fs(xsa) - fs(xs))/h
    return j

def newt2D(fs, xs, tol):
    xsa = la.solve(jac(fs, xs), - fs(xs))
    while(np.any(abs((xsa-xs)/xsa) >= tol)):
        xs = xsa
        hs = la.solve(jac(fs, xs), - fs(xs))
        xsa = xsa + hs
    return xsa


xs = np.array([1.0, 1.0])
xroots = newt2D(fs1, xs, 10**-8)
print("fs1 x Roots Newtonian 2D: ", xroots)
print("fs1(x Roots) = ", fs1(xroots), '\n')

xs = np.array([1.0, 1.0, 1.0])
xroots = newt2D(fs2, xs, 10**-8)
print("fs2 x Roots Newtonian 2D: ", xroots)
print("fs2(x Roots) = ", fs2(xroots), '\n')

# c)
def fs2Jac(xs):
    x0, x1, x2 = xs[0], xs[1], xs[2]
    j = np.empty([3, 3])
    j[0] = np.array([2, -np.cos(x2), x1*np.sin(x2)])
    j[1] = np.array([2*x0, -50*(x1-2), np.cos(x2)])
    j[2] = np.array([7*np.exp(x1), 7*x0*np.exp(x1), -17])
    return j

def newt2Dfs2(fs, xs, tol):
    xsa = la.solve(fs2Jac(xs), - fs(xs))
    while(np.any(abs((xsa-xs)/xsa) >= tol)):
        xs = xsa
        hs = la.solve(fs2Jac(xs), - fs(xs))
        xsa = xsa + hs
    return xsa

xroots = newt2Dfs2(fs2, xs, 10**-8)
print("fs2 x Roots Newtonian 2D (Analytical): ", xroots)
print("fs2(x Roots) (Analytical):", fs2(xroots))

    


    

