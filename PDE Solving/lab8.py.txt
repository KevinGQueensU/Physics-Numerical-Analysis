#%% Q1)
import matplotlib.pyplot as plt
import numpy as np

def plotSet(title, xLabel, yLabel, xStart, xEnd, yStart, yEnd):
    # font size and res
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.dpi'] = 120

    #figure and axis sizes, labels
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_axes([0.1, 0.1, 2, 1])
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xlim(xStart, xEnd)
    ax.set_ylim(yStart, yEnd)

# Function for initial boundary condition
def f(x):
    return 20 + 30 * np.exp(-100 * (x - 0.5)**2)

# Function to solve for temp. over time
def timeSolve(u, dt, dx, diff):
    k = diff*dt/dx**2 # Kappa in Eq. 6
    
    # Checking for Caurant stability
    if dt > (0.5 * dx**2) / diff: 
        print("The time step size is too large for numerical stability.")
        return u
    
    # Looping over the temp values at a time for all positions
    for i in range(1, np.shape(u)[1]):
        u[1:-1, i] = u[1:-1, i-1] + k * (u[2:, i-1] - 2*u[1:-1, i-1] + u[:-2, i-1]) # Eq. 6 from lecture
        
    return u

# Function that sets the initial conditions
def initial(u, xs, f):
    # Boundaries
    u[0, :] = 20
    u[-1, :] = 20
    u[:, 0] = f(xs)
    
    return u

# Input parameters
alpha = 2.3e-4 # diffusion
dx = 0.01
dt = 0.1
L = 1
xs = np.arange(0, L, dx)
ts = np.arange(0, 61, dt)
u = np.zeros((len(xs), len(ts)))

# Solving PDE
u = initial(u, xs, f)
u = timeSolve(u, dt, dx, alpha)

# Plotting for all t's required
plotSet("1D Heat Equation", "Rod Length (m)", r"Temperature $(\degree C)$", -0.05, 1.05, 15, 55)
plt.plot(xs, u[:, 0], label = "t = 0.0s")
plt.plot(xs, u[:, 50], label = "t = 5.0s")
plt.plot(xs, u[:, 100], label = "t = 10.0s")
plt.plot(xs, u[:, 200], label = "t = 20.0s")
plt.plot(xs, u[:, 300], label = "t = 30.0s")
plt.plot(xs, u[:, 600], label = "t = 60.0s")
plt.legend()
plt.show()

#%% Q2 a)
from numpy import linalg as lg

# Function for source
def f(x, y):
    return np.cos(10*x) - np.sin(5*y - np.pi/4)

# Function for initial conditions
def initial(phi):
    # Boundaries
    phi[0, :] = 0  
    phi[-1, :] = 0  
    phi[:, 0] = 0  
    phi[:, -1] = 0
    
    return phi

def jacIter(phi, f, dx, dy, tol):
    # Initializing a meshgred for vectorization of f_pq in Eq. 15
    x = np.arange(0, 2 + dx, dx)
    y = np.arange(0, 1 + dy, dy)
    X, Y = np.meshgrid(x, y)
    
    #Initial norm
    norm1 = lg.norm(phi)
    
    # Looping until the tolerance is met
    while True:
        # Eq. 15 in its entirety using vectorization
        phi[1:-1, 1:-1] = ((phi[1:-1, 2:] + phi[1:-1, :-2]) * dy**2 +(phi[2:, 1:-1] + phi[:-2, 1:-1]) 
                           * dx**2 -dx**2 * dy**2 * np.transpose(f(X, Y)[1:-1, 1:-1])) / (2 * (dx**2 + dy**2))
        
        # Comparing new norm to old, checking if tol. reached
        norm2 = lg.norm(phi)
        if (norm2 - norm1) / norm2 < tol:
            break
        
        norm1 = norm2
        
    # Returning the transpose b/c the meshgrid leds to a transposed matrix, need to revert it back
    return phi.transpose()

# Input parameters
dx = 0.01
dy = 0.01
xs = np.arange(0, 2 + dx, dx)
ys = np.arange(0, 1 + dy, dy)
phi = np.zeros((len(xs), len(ys))) 

# Solving PDE
phi = initial(phi)
phi = jacIter(phi, f, dx, dy, 10e-5/2) # For some reason I can only replicate the plot on the PDF with tol of half of 10e-5

# Plotting countour
plotSet("Poisson Equation", "x", "y", 0.0001, 2, 0, 1) 
plt.imshow(phi, extent=[0, 2, 0, 1], cmap='hot', origin='lower')
plt.colorbar(label=r'$\phi(x, y)$')
plt.show()

#%% Q2 b)

# Function for initial conditions
def initial(phi):
    # New boundaries
    phi[:, 0] = 0.2
    phi[:, -1] = 0.1 
    phi[0, :] = 0.3 
    phi[-1, :] = 0.4 
    
    return phi

# Input parameters
dx = 0.01
dy = 0.01
xs = np.arange(0, 2 + dx, dx)
ys = np.arange(0, 1 + dy, dy)
phi = np.zeros((len(xs), len(ys)))

# Solving PDE
phi = initial(phi)
phi = jacIter(phi, f, dx, dy, 10e-5/2)

# Plotting
plotSet("Poisson Equation", "x", "y", 0.0001, 2, 0, 1) 
plt.imshow(phi, extent=[0, 2, 0, 1], cmap='hot', origin='lower')
plt.colorbar(label=r'$\phi(x, y)$')
plt.show()

#%% BONUS
from numpy import fft as fft

# Function of the spatial period
def f(x, y):
    return np.cos(3*x + 4*y) - np.cos(5*x - 2*y)

# Function that solves Poisson in Fourier Space
def FFTSolve(phi, f, x, y):
    # Initial parameters
    h = x[1] - x[0] # Step size
    n = len(x) # For FT
    if(n % 2 != 0): # CHECKING VALIDITY
        print("Please input even number of points")
        exit(0)
    k = np.arange(0, n, 1) # Coeffecients in Eq. 6
    
    # Vectorization
    X, Y = np.meshgrid(x, y)
    k, l = np.meshgrid(k, k)
    fmesh = f(X, Y) # Computing vectorized function
    fmeshFT = fft.fft2(fmesh) # Computing fft of meshgrid

    # Finding FFT
    phiFT = (fmeshFT*h**2)/((np.cos(2 * np.pi * k/n) + np.cos(2 * np.pi * l/n)  - 2)* 2) # Full Eq. 6
    phiFT[0, 0] = 0 # FIXING DIVIDE BY 0 TERM
    
    # IFT to get solution in real space
    phi = fft.ifft2(phiFT).real  # Only taking the real part
    
    return phi

#Input parameters
dx = 2*np.pi/800
dy = dx
xs = np.arange(0, 2*np.pi, dx)
ys = np.arange(0, 2*np.pi, dy)
phi = np.zeros((len(xs), len(ys)))

# Finding solution
phi = FFTSolve(phi, f, xs, ys)

# Plotting countour
plotSet("Poisson Equation: Periodic Spatial Grid", "x", "y", 0.0001, 2, 0, 2) 
plt.imshow(phi, extent=[0, 2, 0, 2], cmap='hot', origin='lower')
plt.colorbar(label=r'$\phi(x, y)$')
plt.show()

