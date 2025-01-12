#%% Q1 a)
import sympy as sm
import math as m
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci

def plotSet(title, xLabel, yLabel, xStart, xEnd, yStart, yEnd):
    # font size and res
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.dpi'] = 120

    #figure and axis sizes, labels
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 2, 1])
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xlim(xStart, xEnd)
    ax.set_ylim(yStart, yEnd)

# Derivative function
def fun(y, t):
    return -10*y

# Exact solution
def funexact(t):
    return np.exp(-10*t)

# Function from example code
def feval(funcName, *args):
    return eval(funcName)(*args)

# Forward Euler function from example code
def euler(f, y0, t, h): 
    k1 = h*f(y0, t)                     
    y1 = y0+k1
    return y1 

# Backward vectorized Euler function
def backeuler(f, y0, t, h):   
    def f2(y): # Function where Eq. 8 is all moved to one side so I can solve for the roots
        return y - y0 - h * f(y, t)
    y1 = sci.optimize.fsolve(f2, y0) # Use Scipy method to find the roots of previous function, gives y(j+1)
    return y1 


# Stepper function from example code
def odestepper(odesolver, deriv, y0, t):
    y0 = np.asarray(y0)
    y = np.zeros((t.size, y0.size))
    y[0,:] = y0; h=t[1]-t[0]
    y_next = y0

    for i in range(1, len(t)):
        y_next = feval(odesolver, deriv, y_next, t[i-1], h)
        y[i,:] = y_next
    return y

# Initial condition, time array for exact function
y0 = 1
tsExact = np.linspace(0, 0.6, 1000)

# Time array with n = 10, plug into the odestepper with forward and backwards Euler
ts = np.linspace(0, 0.6, 10)
ys1 = odestepper('euler', fun, y0, ts)
ys2 = odestepper('backeuler', fun, y0, ts)

# Plotting all the data for n = 10
plotSet("ODE Stepper: y' = -10t", "t", "y", 0, 0.6, -0.1, 1)
ys1 = odestepper('euler', fun, y0, ts)
ys2 = odestepper('backeuler', fun, y0, ts)
plt.plot(tsExact, funexact(tsExact), 'r', label = "Exact")
plt.plot(ts, ys1, 'bo', label = 'F-Euler (n=10)')
plt.plot(ts, ys2, 'go', label = 'B-Euler (n=10)')
plt.legend()
plt.show()

# Time array with n = 20, plug into the odestepper with forward and backwards Euler
ts = np.linspace(0, 0.6, 20)
ys1 = odestepper('euler', fun, y0, ts)
ys2 = odestepper('backeuler', fun, y0, ts)

# Plotting all the data for n = 20
plotSet("ODE Stepper: y' = -10t", "t", "y", 0, 0.6, -0.1, 1)
plt.plot(tsExact, funexact(tsExact), 'r', label = "Exact")
plt.plot(ts, ys1, 'bo', label = 'F-Euler (n=20)')
plt.plot(ts, ys2, 'go', label = 'B-Euler (n=20)')
plt.legend()
plt.show()

#%% Q1 b)

# Function that seperates the second order differential equation into two separate first order
def fs(y, t):
    y0, y1 = y 
    f0 = -y1 # This is the function for dv/dt = -x(t)
    f1 = y0 # This is the function for dx/dt = v
    return np.array([f0, f1]) 

# Function of the exact solution
def funexact(t):
    return np.cos(t)

# Function that computes the RK4 (vectorized) method with functions f, intial values y0, time t, and step h
def RK4(f, y0, t, h):
    #This section is just Eq. 13, finding k coefficients
    k0 = h*f(y0, t)
    k1 = h*f(y0 + k0/2, t + h/2)
    k2 = h*f(y0 + k1/2, t +h/2)
    k3 = h*f(y0 + k2, t + h)
    
    # Returning the next step
    return  y0 + (1/6) * (k0 + 2*k1 + 2*k2 + k3)

# Defining initial conditions
x0 = 1
v0 = 0
y0s = np.array([v0, x0])

# Defining the time array for dt = 0.01, ysE is the Euler solution, ysRK is the RK4 solution
ts = np.arange(0, 20*np.pi, 0.01)
ysE = odestepper('euler', fs, y0s, ts).transpose()
ysRK = odestepper('RK4', fs, y0s, ts).transpose()
ysExact = funexact(ts)

# Plotting phase space for dt = 0.01, ys[1] corresponds to the position x, and ys[0] corresponds to velocity
plotSet("Phase Space: RK4 vs Euler dt = 0.01", "x", "v", -2, 2, -2, 2)
plt.plot(ysE[1], ysE[0], 'r--', label = 'F-Euler', lw = 5)
plt.plot(ysRK[1], ysRK[0], 'b', label = 'RK4', lw = 5)
plt.legend()
plt.show()

# Plotting x(t) for dt = 0.01, ys[1] corresponds to the position x
plotSet("x(t): RK4 vs Euler dt = 0.01", "t", "x", -1, 20*np.pi + 1, -1.5, 1.5)
plt.plot(ts, ysExact, 'g', label = 'Exact', lw = 5)
plt.plot(ts, ysE[1], 'r--', label = 'F-Euler', lw = 5)
plt.plot(ts, ysRK[1], 'b--', label = 'RK4', lw = 5, dashes = [3, 5])
plt.legend(loc=1)
plt.show()

# Plotting phase space for dt = 0.005, ys[1] corresponds to the position x, and ys[0] corresponds to velocity
ts = np.arange(0, 20*np.pi, 0.005)
ysE = odestepper('euler', fs, y0s, ts).transpose()
ysRK = odestepper('RK4', fs, y0s, ts).transpose()
ysExact = funexact(ts)
plotSet("Phase Space: RK4 vs Euler dt = 0.005", "x", "v", -2, 2, -2, 2)
plt.plot(ysE[1], ysE[0], 'r--', label = 'F-Euler', lw = 5)
plt.plot(ysRK[1], ysRK[0], 'b', label = 'RK4', lw = 5)
plt.legend()
plt.show()

# Plotting x(t) for dt = 0.005, ys[1] corresponds to the position x
plotSet("x(t): RK4 vs Euler dt = 0.005", "t", "y", -1, 20*np.pi + 1, -1.5, 1.5)
plt.plot(ts, ysExact, 'g', label = 'Exact', lw = 5)
plt.plot(ts, ysE[1], 'r--', label = 'F-Euler', lw = 5)
plt.plot(ts, ysRK[1], 'b--', label = 'RK4', lw = 5, dashes = [3, 5])
plt.legend(loc=1)
plt.show()

#%% Q2 a)

# Function that is a slightly altered version of the given code
def odestepper(odesolver, deriv, pos0, t):
    pos0 = np.asarray(pos0) #pos0 now contains [x0, y0, z0]
    pos = np.zeros((t.size, pos0.size)) # pos will contain the x, y, and z coordinates corresponding to [i, 0], [i, 1], [i, 2]
    pos[0,:] = pos0; h=t[1]-t[0]
    pos_next = pos0 

    for i in range(1, len(t)):
        pos_next = feval(odesolver, deriv, pos_next, t[i-1], h)
        pos[i,:] = pos_next
    return pos

# Function of Lorentz Attractor equations that take in the coordinates and time, returns an array of the functions
def fs(x, y, z, t):
    fx = 10*(y-x)
    fy = 28*x-y-x*z
    fz = x*y - (8/3)*z
    return np.array([fx, fy, fz])

# Function that provides the RK4 (vectorized) method, with f functions, pos defining [x, y, z], t time, and h step size
def RK4(f, pos, t, h):
    x, y, z = pos # Retrieving the individual coordinates
    
    # Fairly similar to Eq. 13, finding K coeffecients
    k0 = h*f(x, y, z, t)
    # I take the diagonals here b/c fx should only correspond to k[0], fy to k[1], fz to k[2]. 
    # I dont care about the cross terms such as f(x) to k[1], etc. 
    k1 = h*f(x + k0/2, y + k0/2 , z + k0/2, t + h/2).diagonal() 
    k2 = h*f(x + k1/2, y + k1/2, z + k1/2, t + h/2).diagonal()
    k3 = h*f(x + k2, y + k2, z + k2, t + h).diagonal()
    k = (1/6) * (k0 + 2*k1 + 2*k2 + k3)
    
    # Returning individual coordiantes with their respective k coeffecients
    return  x+k[0], y + k[1], z + k[2] 

# Time array
ts = np.arange(0, 8*np.pi, 0.01)

# Computing the positions for the two separate initial conditions
pos1 = [1, 1, 1.001]
x1, y1, z1 = odestepper('RK4', fs, pos1, ts).transpose() # Taking the transpose b/c the stepper method has position on columns instead of rows
pos2 = [1, 1, 1.]
x2, y2, z2 = odestepper('RK4', fs, pos2, ts).transpose()

# Creating the 3D Plot
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(0, 43)
ax.set_xlabel('x', fontsize = 20)
ax.set_ylabel('y', fontsize = 20)
ax.set_zlabel('z', fontsize=  20)
ax.view_init(azim=20, elev=29)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.title(r"Lorenz Attractor", x = 0.5, y = 1.1)
ax.plot(x1, y1, z1, 'r', label = "[x0, y0, z0] = [1, 1, 1.001]")
ax.plot(x2, y2, z2, 'b--', label = "[x0, y0, z0] = [1, 1, 1.]")
plt.legend(loc=1)
plt.show()

#%% Q2 b)
from matplotlib import animation

# Creating the 3D Plot
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(0, 43)
ax.set_xlabel('x', fontsize = 20)
ax.set_ylabel('y', fontsize = 20)
ax.set_zlabel('z', fontsize=  20)
ax.view_init(azim=20, elev=29)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.title(r"Lorenz Attractor", x = 0.5, y = 1.02, fontsize=20)
plt.legend(['[x0, y0, z0] = [1, 1, 1.001]', '[x0, y0, z0] = [1, 1, 1.]'], loc=1)

# Two separate animation lines, corresponding to the two initial conditions
line1, = ax.plot([], [], [], 'r', lw=1)
line2, = ax.plot([], [], [], 'b--', lw=1)

# Function to update animation, (x, y, z) correspond to (x1, y1, z1), (xi, yi, zi) correspond to (x2, y2, z2)
def update(i, x, y, z, xi, yi, zi):
    # I was looking at matplotlib tutorials and this is similiar to how they did it, idk whats going on though.
    line1.set_data(x[0:i], y[0:i])
    line1.set_3d_properties(z[0:i])
    
    line2.set_data(xi[0:i], yi[0:i])
    line2.set_3d_properties(zi[0:i])
    return

# Defining and showing the animation.
ani = animation.FuncAnimation(fig, update, frames = 10000, fargs=(x1, y1, z1, x2, y2, z2), interval=0)
plt.show()