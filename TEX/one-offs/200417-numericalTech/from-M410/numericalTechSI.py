"""
File meant to show numerical integration applied via python
should essentially be the same as the matlab file(s)

Structured in a way that is more related to the simulation method in PSLTDSim

TODO: Include solve_ivp comparison
TODO: include trapezoidal integration comparisons

lambda is the python equivalent of matlab anonymous functions
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Method Definitions
def euler(fp, x0, y0, ts):
    """
    fp = Some derivative function of x and y
    x0 = Current x value
    y0 = Current y value
    ts = time step
    Returns y1 using Euler or tangent line method
    """    
    return y0 + fp(x0,y0)*ts

def adams2(fp, x0, y0, xN, yN, ts):
    """
    fp = Some derivative function of x and y
    x0 = Current x value
    y0 = Current y value    
    xN = Previous x value
    yN = Previous y value
    ts = time step
    Returns y1 using Adams-Bashforth two step method
    """
    return y0 + (1.5*fp(x0,y0) - 0.5*fp(xN,yN))*ts 

def rk45(fp, x0, y0, ts):
    """
    fp = Some derivative function of x and y
    x0 = Current x value
    y0 = Current y value
    ts = time step
    Returns y1 using Runge-Kutta method
    """
    k1 = fp(x0, y0)
    k2 = fp(x0 +ts/2, y0+ts/2*k1)
    k3 = fp(x0 +ts/2, y0+ts/2*k2)
    k4 = fp(x0 +ts, y0+ts*k3)    
    return y0 + ts/6*(k1+2*k2+2*k3+k4)


caseName = 'Sinusodial Example'
tStart =0
tEnd = 1
numPoints = 4
ic = [0,0] # initial condition x,y
fp = lambda x, y: -2*np.pi*np.cos(2*np.pi*x)
f = lambda x,c: -np.sin(2*np.pi*x)+c
findC = lambda x,y: y+2*np.pi*np.sin(2*np.pi*x)


# Find C from integrated equation for exact soln
c = findC(ic[0], ic[1])

# Initialize current value dictionary
# Shown to mimic PSLTDSim record keeping
cv={
        't' :ic[0],
        'yE': ic[1],
        'yRK': ic[1],    
        'yAB': ic[1],
        'ySI': ic[1],
        }

# Calculate time step
ts = (tEnd-tStart)/numPoints

# Calculate exact solution
tExact = np.linspace(tStart,tEnd, 100)
yExact = f(tExact, c)

# Initialize running value lists
t=[]
ySI = []
tSI = []

t.append(cv['t'])
#tSI.append(cv['t'])
#ySI.append(cv['ySI'])

# Start Simulation
while cv['t']< tEnd:


    # Runge-Kutta via solve IVP...
    soln = solve_ivp(fp, (cv['t'], cv['t']+ts), [cv['ySI']])

    # cat list soln
    ySI += list(soln.y[-1])#[1:-1]
    tSI += list(soln.t)#[1:-1]
    # ensure correct cv
    cv['ySI'] = ySI[-1]

    cv['t'] += ts
    t.append(cv['t'])

# Generate Plots
fig, ax = plt.subplots()

ax.set_title('Approximation Comparison\n' + caseName)
ax.plot(tExact,yExact, c=[0,0,0], linewidth=2, label="Exact")
ax.plot(tSI,ySI,marker='d', markersize=10, fillstyle='none')
#    ax.plot(t,yE, marker='o', fillstyle='none', linestyle=':', c=[0.7,0.7,0.7], label="Euler")
#    ax.plot(t,yRK, marker='*', markersize=10, fillstyle='none',linestyle=':', c=[1,0,1], label="RK45")
#    ax.plot(t,yAB, marker='s', fillstyle='none',linestyle=':', c =[0,1,0], label="AB2")

fig.set_dpi(150)
fig.set_size_inches(9/3, 2.5)
ax.set_xlim(min(t), max(t))
ax.grid(True)
ax.legend(loc='best',  ncol=2, fontsize='x-small' )
fig.tight_layout()    
plt.show(block = True)
plt.pause(0.00001)





