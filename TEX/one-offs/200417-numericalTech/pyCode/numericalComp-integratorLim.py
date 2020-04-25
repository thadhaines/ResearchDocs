"""
File meant to show Python numerical integration applied via scipy functions
Structured in a way that is more related to the simulation method in PSLTDSim

lambda is the python equivalent of matlab anonymous functions
"""
# Package Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal

# step input Integrator example
tStart =0
tEnd = 20
numPoints = 20*2

maxLim = 2
minLim = -2

initState = 0
ic = [0,initState] # initial condition x,y

system = signal.lti([1],[1,0])

# Initialize current value dictionary
# Shown to mimic PSLTDSim record keeping
cv={
    't' :ic[0],
    'yLS': ic[1], 
    'yLSlim': ic[1],
    }

# Calculate time step
ts = (tEnd-tStart)/numPoints

# Initialize running value lists
t=[]

# lsim
yLS = []
xLS = [] 
yLSlim = []
xLSlim = [] 

# running history
r_U = [1]

t.append(cv['t'])
yLS.append(cv['yLS'])
xLS.append(cv['yLS'])
yLSlim.append(cv['yLS'])
xLSlim.append(cv['yLS'])

# Start Simulation
while cv['t']< tEnd:

    if cv['t'] < 5:
        U = 1
    elif cv['t'] < 15:
        U = -1
    elif cv['t'] < 18:
        U = 1
    else:
        U = 0

    # lsim solution
    if cv['t'] > 0:
        tout, ylsim, xlsim = signal.lsim(system, [U,U], [0,ts], xLS[-1])
        tout, ylsimLIM, xlsimLIM = signal.lsim(system, [U,U], [0,ts], xLSlim[-1])
    else:
        tout, ylsim, xlsim = signal.lsim(system, [U,U], [0,ts], initState)
        tout, ylsimLIM, xlsimLIM = signal.lsim(system, [U,U], [0,ts], initState)
        
    # handle lsim output data
    cv['yLS']=ylsim[-1]
    if cv['yLS'] > maxLim:
        cv['yLS']= maxLim
    if cv['yLS'] < minLim:
        cv['yLS'] = minLim
          
    yLS.append(cv['yLS'])
    xLS.append(xlsim[-1]) # this is the state

    # handle limited lsim output data
    cv['yLSlim']=ylsimLIM[-1]
    if cv['yLSlim'] > maxLim:
        cv['yLSlim'] = maxLim
        xlsimLIM = [maxLim]
    elif cv['yLSlim'] < minLim:
        cv['yLSlim'] = minLim
        xlsimLIM = [minLim]

    yLSlim.append(cv['yLSlim'])
    xLSlim.append(xlsimLIM[-1]) # this is the state
    
    # Increment and log time
    cv['t'] += ts
    t.append(cv['t'])
    r_U.append(U)


# Generate Plot
fig, ax = plt.subplots()
ax.set_title('Integrator Wind Up Example')

#Plot all lines

ax.plot(t,yLS,
        #marker='x',
        markersize=10,
        fillstyle='none',
        linestyle='--',
        c=[0.6,0.6,0.6],
        label="BLy")
ax.plot(t,xLS,
        marker='o',
        markersize=5,
        fillstyle='none',
        linestyle='',
        c=[0.6,0.6,0.6],
        label="BLx") 
ax.plot(t,yLSlim,
        #marker='+',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c =[1,0,1],
        label="CLy")
ax.plot(t,xLSlim,
        marker='s',
        markersize=5,
        fillstyle='none',
        linestyle='',
        c =[1,0,1],
        label="CLx")
ax.step(t,r_U,
        #marker='s',
        #markersize=5,
        #fillstyle='none',
        #linestyle=':',
        c =[0,0,0],
        label="Input")

# Format Plot
fig.set_dpi(150)
fig.set_size_inches(9, 2.5)
ax.set_xlim(min(t), max(t))
ax.grid(True, alpha=0.25)
ax.legend(loc='best',  ncol=3)
fig.tight_layout()    
plt.show(block = True)
plt.pause(0.00001)
