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
    plt.rcParams['figure.dpi'] = 120

    #figure and axis sizes, labels
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 2, 1])
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xlim(xStart, xEnd)
    ax.set_ylim(yStart, yEnd)


#%% Q1 a)
# Defining the function
def f(t, a0, a1, a2, w0, w1, w2):
    return a0*np.sin(w0*t) + a1*np.sin(w1*t) + a2*np.sin(w2*t)

# Discrete Fourier Transform, takes in yj points and n spacing
def DFT(yj, n):
    if(n % 2 != 0):
        print("Please input even number of points")
        exit(0)
    k = np.arange(0, n, 1) # Array of integer values from 0 to n-1
    ck = np.exp(-2*np.pi*1j*k*k[:, None]/n) # 2D array to represent the coeffecients in Eq. 12
    return np.matmul(yj, ck) # Taking the product of the yj points and coeffecients to return the DFT
    
# Inverse Discrete Fourier Transform, takes in yk points and n spacing
def IDFT(yk, n):
    if(n % 2 != 0):
        print("Please input even number of points")
        exit(0)
    j = np.arange(0, n, 1) # Array of integer values from 0 to n-1
    ck = np.exp(2*np.pi*1j*j*j[:, None]/n) # 2D array to represent the coeffecients in Eq. 13
    return np.matmul(yk, ck) * 1/n # Taking the product of the yj points and coeffecients to return the IDFT

# Settings for n = 30
n1 = 30.
dt1 = 2*np.pi/n1   # Spacing between each time
ts1 = np.arange(0., 2*np.pi, dt1)
ys1 = f(ts1, 3., 1., 0.5, 1., 4., 7.)

# Settings for n = 60
n2 = 60.
dt2 = 2*np.pi/n2
ts2 = np.arange(0., 2*np.pi, dt2)
ys2 = f(ts2, 3., 1., 0.5, 1., 4., 7.)

# Plotting the original functions
plotSet(r"$y(t) = 3sin(t) + sin(4t) + 0.5sin(7t)$", "t", "y(t)", -0.2, 6.5, -4.5, 4.5)
plt.plot(ts2, ys2, 'b', lw = 3, label = "n = 60")
plt.plot(ts1, ys1, 'r--', lw = 3, label = "n = 30")
plt.legend()
plt.show()

# Fourier Transform of n = 30
yjs1 = DFT(ys1, n1)
dw1 = 2*np.pi/(n1*dt1) # Spacing between each frequency
ws1 = np.arange(0.0, dw1*n1, dw1)

# Fourier Transform of n = 60
yjs2 = DFT(ys2, n2)
dw2 = 2*np.pi/(n2*dt2)
ws2 = np.arange(0.0, dw2*n2, dw2)

# Plotting Fourier Transforms
plotSet("Fourier Transform: DFT w/ np.arange", r"$\omega$", r'$|\tilde{y}|$', -1, 60, -5, 95)
plt.stem(ws2, abs(yjs2), 'b', markerfmt=" ", basefmt="b", label="n = 60")
plt.stem(ws1, abs(yjs1), 'r--', markerfmt=" ", basefmt="r--", label="n = 30")
plt.legend()
plt.show()

#%% Q1 b)
# Similar to Q1 a), except using linspace instead of arange when creating time and frequency ranges
n1 = 30
ts1 = np.linspace(0., 2*np.pi, n1)
dt1 = ts1[1]-ts1[0]
ys1 = f(ts1, 3., 1., 0.5, 1., 4., 7.)
yjs1 = DFT(ys1, n1)
dw1 = 4*np.pi/(n1*dt1)
maxw1 = 4*np.pi/(2*dt1) # The maximum frequency/Nyquist frequency (multiplying by 4 b/c t range is from 0 to 2pi not 0 to pi)
ws1 = np.linspace(0., maxw1, n1)

n2 = 60
ts2 = np.linspace(0., 2*np.pi, n2)
dt2 = ts2[1]-ts2[0]
ys2 = f(ts2, 3., 1., 0.5, 1., 4., 7.)
yjs2 = DFT(ys2, n2)
dw2 = 2*np.pi/(n2*dt2)
maxw2 = 4*np.pi/(2*dt2)
ws2 = np.linspace(0., maxw2, n2)

plotSet("Fourier Transform: DFT w/ np.linspace", r"$\omega$", r'$|\tilde{y}|$', -1, 60, -5, 95)
plt.stem(ws2, abs(yjs2), 'b', markerfmt=" ", basefmt="b", label="n = 60")
plt.stem(ws1, abs(yjs1), 'r--', markerfmt=" ", basefmt="r--", label="n = 30")
plt.legend()
plt.show()

# The F.T graph looks wonky because linspace is inclusive of the stopping value, whereas arange is not

#%% Q1 c)
# Settings for n = 30
n = 60.
dt = 2*np.pi/n
ts = np.arange(0., 2*np.pi, dt)
ys = f(ts, 3., 1., 0.5, 1., 4., 7.)

# Taking the Discrete Fourier Transform
yjs = DFT(ys, n)
dw = 2*np.pi/(n*dt)
maxw = 2*np.pi/(2*dt)
ws = np.arange(0.0, dw*n, dw)

# Taking the Inverse Discrete Fourier transform 
yks = IDFT(yjs, n)

# Plotting original function and Inverse Fourier Transform
plotSet("Fourier Transform: y(t) and IDFT", "t", r"$\Re(y)$", -0.2, 6.5, -4.5, 4.5)
plt.plot(ts, ys, 'b', label = "IDFT", lw=3)
plt.plot(ts, np.real(yks), 'r--',  label = "y(t)", lw=2)
plt.legend()
plt.show()

#%% Q2
# Defining the gaussian function
def gauss(t, sig, wp):
    return np.exp((-t**2.)/(sig**2.)) * np.cos(wp*t)

# Plotting the original function
n = 60.
sigma = 0.5
w = 0.
dt = 2*np.pi/n
ts = np.arange(-np.pi, np.pi, dt)
ys = gauss(ts, sigma, w)
plotSet("Gaussian Pulse", "t", "y(t)", -3.5, 3.5, -0.1, 1.1)
plt.plot(ts, ys, 'b')
plt.show()

# Taking the Discrete Fourier Transform
yjs = DFT(ys, n)
dw = 2*np.pi/(n*dt)
ws = np.arange(0.0, dw*n, dw)

# Shifting the Fourier Transform using numpy (taken directly from lectures)
ws_shift = np.fft.fftfreq(len(ts), dt)*2.*np.pi
ws_shift = np.fft.fftshift(ws_shift)
yjs_shift = np.fft.fftshift(yjs)

plotSet("Fourier Transform: Gaussian Pulse", r"$\omega$", r"$|\tilde{y}|$", -39, 59.9, -0.1, 9)
plt.plot(ws, np.abs(yjs), 'b', lw = 3, label = "Unshifted")
plt.plot(ws_shift, np.abs(yjs_shift), 'r--', lw = 3, label = "Shifted")
plt.legend()
plt.show()

#%% Q3

# Settings for n = 200
n = 200.
dt = 8*np.pi/n
ts = np.arange(0, 8*np.pi, dt)
ys = f(ts, 3., 1., 0., 1., 10., 0.)

# Taking the Discrete Fourier Transform
yjs = DFT(ys, n)
dw = 2*np.pi/(n*dt)
maxw = 2*np.pi/(2*dt)
ws = np.arange(0.0, dw*n, dw)

# Finding any pulses in the Fourier Transform, and then setting those values to 0
yjsfil = np.where((np.abs(yjs) > 0.), 0., yjs)

# It looks like in the lab PDF that only the first pulse is considered, so I set the first 20 values back to the original Fourier Transform
yjsfil[0:20] = yjs[0:20]

# Plotting filter and unfiltered for F.T
plotSet("Fourier Transform: Filtering", "w", "|y|", -0.5, 50, -10, 310)
plt.plot(ws, np.abs(yjsfil), 'b', lw = 3, label = "Filtered")
plt.plot(ws, np.abs(yjs), 'r--', lw = 2, label = "Unfiltered")
plt.legend()
plt.show()

# Taking the Inverse Discrete Fourier Transform of the filtered data
ysfil = IDFT(yjsfil, n)

# Plotting unfiltered and filtered functions
plotSet("Filtered vs. Unfiltered Function", "t", "y(t)", -0.5, 26, -4.5, 4.5)
plt.plot(ts, ysfil, 'b', lw=3, label = "Filtered")
plt.plot(ts, ys, 'r--', lw=3, label = "Unfiltered")
plt.legend()
plt.show()

# The reason the heights are wrong is because the impulse frequency at the end of the F.T is being ignored
# This gives the incorrect because it is NOT noise

# Correct filtered version
yjsfilfixed = np.where((np.abs(yjs) > 0) & (np.abs(yjs) < 150), 0, yjs)

# Taking the Inverse Discrete Fourier Transform of the filtered data
ysfilfixed = IDFT(yjsfilfixed, n)

# Plotting filter and unfiltered for F.T
plotSet("Fourier Transform: Filtering (Fixed)", "w", "|y|", -0.5, 50, -10, 310)
plt.plot(ws, np.abs(yjsfilfixed), 'b', lw = 3, label = "Filtered")
plt.plot(ws, np.abs(yjs), 'r--', lw = 2, label = "Unfiltered")
plt.legend()
plt.show()

# Plotting unfiltered and filtered functions
plotSet("Filtered vs. Unfiltered Function (Fixed)", "t", "y(t)", -0.5, 26, -4.5, 4.5)
plt.plot(ts, ysfilfixed, 'b', lw=3, label = "Filtered")
plt.plot(ts, ys, 'r--', lw=3, label = "Unfiltered")
plt.legend()
plt.show()

