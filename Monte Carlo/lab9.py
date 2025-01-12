#%% Q1 a)
import sympy as sm
import matplotlib.pyplot as plt
import numpy as np
def plotSet(title, xLabel, yLabel, xStart, xEnd, yStart, yEnd):
    # font size and res
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.dpi'] = 120

    #figure and axis sizes, labels
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_axes([0.1, 0.1, 2, 1])
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xlim(xStart, xEnd)
    ax.set_ylim(yStart, yEnd)
    
# Function that computes pi using monte carlo
def MCPi(n, r):
    rand = np.random.uniform(-r, r, (2, n)) # Getting a random coordinate (x, y) within the rxr square
    pi = rand[:, np.linalg.norm(rand, axis=0) <= r] # Filtering all values that are within the circle, do this by computing norm of each coordinate
    pi = len(pi[0])*(1/r)**2/n # Approximating pi
    return rand, pi

# Input values
n = 1000
r = 0.5

# Computing values
rand, pi = MCPi(1000, r)

# Plotting MC approx.
plotSet("Monte Carlo: " + r"$\pi \approx$" + "{:.4f}".format(pi) + ", n = " + str(n), "x", "y", -0.599, 0.6, -0.6, 0.6)
circle1 = plt.Circle((0, 0), r, color='r') # Adding a circle
plt.gca().add_patch(circle1) # Plotting circle
plt.plot(rand[0], rand[1], 'bo')
plt.show()

#%% Q1 b)
# Function of e^x
def f(x):
    return np.exp(x)

# Function that uses sympy to integrate from a to b for e^x
def exact(a, b):
    x = sm.symbols('x')
    return sm.integrate(sm.exp(x), (x, a, b))

# Function that uses M.C to approximate an integral between a and b with n points for function f
def MCI(a, b, n, f):
    pts = np.random.uniform(a, b, int(n)) # Generating points
    f = f(pts) 
    return 1/n * np.sum(f) 

# Input values
N = np.logspace(2, 6, 500) # log space as required by pdf
approx = np.zeros(500)

# Iterating over all N values
for i in range(500):
    approx[i] = MCI(0, 1, N[i], f) # I couldn't get np.random.uniform to vectorize
exact = exact(0, 1)

# PLotting
plotSet("Monte Carlo: " + r"$I = {e^x}$", "N", "I", 10**2, 10**6, 1.56, 1.8)
plt.xscale("log")
plt.plot(N, approx, 'ro', label = 'M.C')
plt.axhline(y = exact, label = 'Exact', lw = 5)
plt.legend()

#%% Q2
import numpy as np
import matplotlib.pyplot as plt

# Function to intialize plot altered so I could plot two graphs on one figure
def plotsSet(title, xLabel, yLabel, xStart, xEnd, yStart, yEnd):
    # font size and res
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.dpi'] = 120

    # figure and axis sizes, labels
    fig = plt.figure(figsize=(20, 20))  # Increase height to accommodate two plots
    ax1 = fig.add_axes([0.1, 0.55, 0.8, 0.4])  # Top plot
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])  # Bottom plot
    plt.suptitle(title)

    ax2.set_xlabel(xLabel[0])
    ax1.set_ylabel(yLabel[0])
    ax1.set_ylim(yStart[0], yEnd[0])
    ax2.set_xlim(xStart[0], xEnd[0])

    ax2.set_xlabel(xLabel[1])
    ax2.set_ylabel(yLabel[1])
    ax2.set_xlim(xStart[1], xEnd[1])
    ax2.set_ylim(yStart[1], yEnd[1])

    ax1.set_xticks([])
    return ax2, ax1

# Function that initializes the spin, energy, and magentization values. (Provided by the lecture slides)
def initialize(N, p):
    spin = np.ones(N)
    E = 0. ; M = 0.
    for i in range (1, N):
        if np.random.rand(1) < p:
            spin[i] = -1
        E = E - spin[i - 1] * spin[i] # Energy
        M = M + spin[i] # Magnetization
    # periodic bc inclusion
    E = E - spin[N - 1] * spin[0]
    M = M + spin[0]
    return spin, E, M

# Function that updates the spin, energy, and magentization values. (Provided by the lecture slides)
def update(N, spin, kT, E, M) :
    num = np.random.randint(0 , N - 1)
    flip = 0
    # periodic bc returns 0 if i + 1 == N , else no change :
    dE = 2 * spin[num] * (spin[num - 1] + spin[(num + 1) % N])
    # if dE is negative , accept flip :
    if dE < 0.0:
        flip = 1
    else :
        p = np.exp(-dE / kT)
        if np.random.rand() < p:
            flip = 1
    # otherwise , reject flip
    if flip == 1 :
        E += dE
        M -= 2 * spin[num]
        spin[num] = -spin[num]
    return E , M

# Funcion that simulates the spin evolution for NMc iterations
def simulate(NMc, spin, kT, E, M):
    N = len(spin) # N value for amount of spins
    
    # Creating arrays to hold the previous values for each iteration
    spins = np.zeros((NMc + 1, N)) 
    energies = np.zeros(NMc +1)
    magnetizations = np.zeros(NMc + 1)
    NMcs = np.arange(NMc + 1)
    
    # Copying initial values
    spins[0] = spin.copy()
    energies[0] = E
    magnetizations[0] = M

    # Iterating all values using update function
    for i in range(NMc):
        E, M = update(N, spin, kT, E, M) # Update energy and magnetism
        spins[i+1] = spin # Add the new spins
        energies[i+1] = E # Add the new energy
        magnetizations[i+1] = M # Add the new magnetization

    return spins, energies, magnetizations, NMcs

# Function that returns the expected energy level at a given temperature
def expected(N, kT):
    return -N*np.tanh(1/kT) # Equation from lecture slides
    
# Input values
N = 50 # number of spins
p = 0.6 # order parameter
kT1 = 0.1 # temps
kT2 = 0.5
kT3 = 1.0
total = 40*N # number of monte carlo iterations

print("Spin down is blue, spin up is yellow \n")
# Simulating evolutions for each value of kT, and calculating expected values
spin, E, M = initialize(N, p)
spins1, energies1, magnetizations1, iterations = simulate(total, spin, kT1, E, M)
energyExp1 = expected(N, kT1)
spin, E, M = initialize(N, p)
spins2, energies2, magnetizations2, _ = simulate(total, spin, kT2, E, M)
energyExp2 = expected(N, kT2)
spin, E, M = initialize(N, p)
spins3, energies3, magnetizations3, _ = simulate(total, spin, kT3, E, M)
energyExp3 = expected(N, kT3)

# Plotting spin evolutions
plt1, plt2 = plotsSet(r"Spin Evolution: $k_BT = 0.1$", ["_", "Iteration/N"], ["N Spins", r"$Energy/N\epsilon$"], [0, 0], [total, total/N], [0.0, -1.1], [N-1, max(energies1/N)])
plt1.plot(iterations/N, energies1/N, 'b', label=r"E")
plt1.axhline(energyExp1/N, 0, 40, color = 'r', label = r'$< E >_{an}$')
plt2.imshow(spins1.transpose(), cmap='plasma', aspect='auto')
plt.legend()
plt.show()

plt1, plt2 = plotsSet(r"Spin Evolution: $k_BT = 0.5$", ["_", "Iteration/N"], ["N Spins", r"$Energy/N\epsilon$"], [0, 0], [total, total/N], [0.0, -1.12], [N-1, max(energies2/N)])
plt1.plot(iterations/N, energies2/N, 'b', label=r"E")
plt1.axhline(energyExp2/N, 0, 40, color = 'r', label = r'$< E >_{an}$')
plt2.imshow(spins2.transpose(), cmap='plasma', aspect='auto')
plt.legend()
plt.show()

plt1, plt2 = plotsSet(r"Spin Evolution: $k_BT = 1.0$", ["_", "Iteration/N"], ["N Spins", r"$Energy/N\epsilon$"], [0, 0], [total, total/N], [0.0, -1.15], [N-1, max(energies3/N)])
plt1.plot(iterations/N, energies3/N, 'b', label=r"E")
plt1.axhline(energyExp3/N, 0, 40, color = 'r', label = r'$< E >_{an}$')
plt2.imshow(spins3.transpose(), cmap='plasma', aspect='auto')
plt.legend()
plt.show()

print("Choosing order parameters that result in more parallel spins such as p = 0.9 or p = 0.1 leads to the energy quickly flatlining for the lower KbT values, but the higher value of 1.0 still fluctuates around the expected value.")
print("\nThe initial energy value magnitudes are also increased as opposed to a more warm start. To see this in my code, just change p closer to 1 or 0.")

#%% Q3
from numba import jit

print("Could not figure out how to vectorize this, so it takes a while to compute the values. Change NMc to be lower if not interested in accuracy.")
# Function that returns the average expected value at a kT point
@jit(nopython=True)
def averageSim(NMc, N, kT, p):
    Es, Ms = np.empty(NMc), np.empty(NMc)
    
    # Initial states
    spin, E, M = initialize(N, p)
    
    # Simulates NMc times.
    spins, Es, Ms, _ = simulate(NMc, spin, kT, E, M)
    
    # Taking last data points and simulate again NMc times.
    _, Es, Ms, _ = simulate(NMc, spin, kT, Es[-1], Ms[-1])
    
    # Returning the averages
    avgE = np.sum(Es)/NMc
    avgM = np.sum(Ms)/NMc
    
    return avgE, avgM

# Input values
N = 50 # number of spins
p = 0.6 # order parameter
NMc = 160*N # number of Monte Carlo iterations
kTN = 100 # number of temperatures
kTs = np.linspace(0.1, 6, kTN)
avgE = np.empty(kTN)
avgM = np.empty(kTN)
expE = np.empty(kTN)
expM = np.empty(kTN)

# Iterate for all kT
for i in range(kTN):
    avgE[i], avgM[i] = averageSim(NMc,  N, kTs[i], p)
    expE[i] = expected(N, kTs[i])
    expM[i] = 0
    
# Plot
plotSet("Energy vs. Temperature", r"$kT/\epsilon$", r"<E>/N$\epsilon$", 0, 6, -1.1, 0)
plt.plot(kTs, avgE/N, 'bo', label = "Monte Carlo") 
plt.plot(kTs, expE/N, 'r', label = "Expected")
plt.legend()
plt.show()
plotSet("Magnetization vs. Temperature", r"$kT/\epsilon$", r"<M>/N$\epsilon$", 0, 6, -2, 2)
plt.plot(kTs, avgM/N, 'bo', label = "Monte Carlo")
plt.plot(kTs, expM/N, 'r', label = "Expected")
plt.legend()
plt.show()

# Vectorization would save a lot of computing time.

    
    
    
    