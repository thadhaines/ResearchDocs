"""
File meant to show Python numerical integration applied via scipy functions
Focus on multistage system
Structured in a way that is more related to the simulation method in PSLTDSim

lambda is the python equivalent of matlab anonymous functions
"""
# Package Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal

# Method Definitions
def trapezoidalPost(x,y):
    """
    x = list of x values
    y = list of y values
    Returns integral of y over x.
    Assumes full lists / ran post simulation
    """
    integral = 0
    for ndx in range(1,len(x)):
        integral+= (y[ndx]+y[ndx-1])/2 * (x[ndx]-x[ndx-1])
    return integral


# step multi order system
tStart =0
tEnd = 7
numPoints = 7*2
blkFlag = True # for holding plots open

U = 1
T0 = 0.4
T2 = 45
T1 = 5
T3 = -1
T4 = 0.5

# Combined system
alphaNum = (T1*T3)
alphaDen = (T0*T2*T4)
alpha = alphaNum/alphaDen

num = alphaNum*np.array([1, 1/T1+1/T3, 1/(T1*T3)])
den  = alphaDen*np.array([1, 1/T4+1/T0+1/T2, 1/(T0*T4)+1/(T2*T4)+1/(T0*T2),
                 1/(T0*T2*T4)])

system = signal.lti(num,den)

# Staged system
sys1 = signal.StateSpace([-1.0/T0],[1.0/T0],[1.0],0.0)
sys2 = signal.StateSpace([-1.0/T2],[1.0/T2],[1.0-T1/T2],[T1/T2])
sys3 = signal.StateSpace([-1.0/T4],[1.0/T4],[1.0-T3/T4],[T3/T4])

# Experimental System
sysExp = sys2*sys3

# Exact solution maths
# PFE
A = ((1/T1-1/T0)*(1/T3-1/T0))/((1/T2-1/T0)*(1/T4-1/T0))
B = ((1/T1-1/T2)*(1/T3-1/T2))/((1/T0-1/T2)*(1/T4-1/T2))
C = ((1/T1-1/T4)*(1/T3-1/T4))/((1/T0-1/T4)*(1/T2-1/T4))

initState = 0 # for steady state start
ic = [0,0] # initial condition x,y
fp = lambda x, y: alpha*(A*np.exp(-x/T0)+B*np.exp(-x/T2)+C*np.exp(-x/T4))
f = lambda x, c: alpha*(-T0*A*np.exp(-x/T0)-T2*B*np.exp(-x/T2)-T4*C*np.exp(-x/T4))+c
findC = lambda x, y : alpha*(A*T0+B*T2+C*T4)

c = findC(ic[0], ic[1])
calcInt = (
    alpha*A*T0**2*np.exp(-tEnd/T0) +
    alpha*B*T2**2*np.exp(-tEnd/T2) +
    alpha*C*T4**2*np.exp(-tEnd/T4) +
    c*tEnd -
    alpha*(A*T0**2+B*T2**2+C*T4**2)

    )# Calculated integral
    
# Initialize current value dictionary
# Shown to mimic PSLTDSim record keeping
cv={
    't' :ic[0],
    'yCombined': ic[1],
    'yExp' : ic[1],
    'y1': ic[1],
    'y2' : ic[1],
    'y3' : ic[1],
    }

# Calculate time step
ts = (tEnd-tStart)/numPoints

# Initialize running value lists
t=[]
t.append(cv['t'])

# Output lists
# combined
yCombined = [cv['yCombined']]
xCombined = [np.array([cv['yCombined'],cv['yCombined'],cv['yCombined']])] # required to track state history

# staged
y1= [cv['y1']]
y2 = [cv['y2']]
y3 = [cv['y3']]
x1 = [cv['y1']]
x2 = [cv['y1']]
x3 = [cv['y1']]

# experimental
yExp = [cv['yExp']]
xExp = [np.array([cv['y1'],cv['y1']])] # required to track state history


# Find C from integrated equation for exact soln
c = findC(ic[0], ic[1])

# Calculate exact solution
tExact = np.linspace(tStart,tEnd, 1000)
yExact = f(tExact, c)

# Start Simulation
while cv['t']< tEnd:

    # lsim combined
    if cv['t'] > 0:
        tout, ylsim, xlsim = signal.lsim(system, [U,U], [0,ts], xCombined[-1])
    else:
        tout, ylsim, xlsim = signal.lsim(system, [U,U], [0,ts], initState)

    # lsim staged
    if cv['t'] > 0:
        tout, y1Sln, x1Sln = signal.lsim(sys1, [U,U], [0,ts], x1[-1])
        tout, y2Sln, x2Sln = signal.lsim(sys2, [y1Sln[-1],y1Sln[-1]], [0,ts], x2[-1])
        tout, y3Sln, x3Sln = signal.lsim(sys3, [y2Sln[-1],y2Sln[-1]], [0,ts], x3[-1])
    else:
        tout, y1Sln, x1Sln = signal.lsim(sys1, [U,U], [0,ts], initState)
        tout, y2Sln, x2Sln = signal.lsim(sys2, [y1Sln[-1],y1Sln[-1]], [0,ts], initState)
        tout, y3Sln, x3Sln = signal.lsim(sys3, [y2Sln[-1],y2Sln[-1]], [0,ts], initState)

    # lsim experimental
    if cv['t'] > 0:
        tout, yExpSln, xExpSln = signal.lsim(sysExp, [y1Sln[-1],y1Sln[-1]], [0,ts], xExp[-1])
    else:
        tout, yExpSln, xExpSln = signal.lsim(sysExp, [y1Sln[-1],y1Sln[-1]], [0,ts], initState)

    # handle combined lsim output data
    cv['yCombined']=ylsim[-1]
    yCombined.append(cv['yCombined'])
    xCombined.append(xlsim[-1]) # this is the state
                                   
    # handle staged lsim output data
    cv['y1']= y1Sln[-1]
    cv['y2']= y2Sln[-1]
    cv['y3']= y3Sln[-1]
                                         
    y1.append(cv['y1'])
    y2.append(cv['y2'])
    y3.append(cv['y3'])
                                         
    x1.append(x1Sln[-1])
    x2.append(x2Sln[-1]) 
    x3.append(x3Sln[-1])

    # handle experimental system output
    cv['yExp'] = yExpSln[-1]
    yExp.append(cv['yExp'])
    xExp.append(xExpSln[-1])
                                         
    # Increment and log time
    cv['t'] += ts
    t.append(cv['t'])

# Generate Plot
fig, ax = plt.subplots()
ax.set_title('Effect of Dynamic Staging')

#Plot all lines
ax.plot(tExact,yExact,
        c = [0.6, 0.6, 0.6],
        linewidth=2,
        label="Exact")
ax.plot(t,yCombined,
        marker='o',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c=[0,0,0],
        label="One Stage")
ax.plot(t,yExp,
        marker='+',
        markersize=10,
        fillstyle='none',
        c = [1,0,1],
        linestyle=':',
        label="Two Stages")
ax.plot(t,y3,
        marker='^',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c = [0,1,0],
        label="Three Stages")


# Format Plot
fig.set_dpi(150)
fig.set_size_inches(9, 2.5)
ax.set_xlim(min(t), max(t))
ax.grid(True, alpha=0.25)
ax.legend(loc='lower right',  ncol=1)
fig.tight_layout()    
plt.show(block = False)
plt.pause(0.00001)

"""
# Generate State Plot
fig, ax = plt.subplots()
ax.set_title('State Comparisons')
ss = system._as_ss()
C = ss.C
xCombined = np.array(xCombined)*C # multiply by c xExp = np.array(xExp)

ax.plot(t,xCombined[:,0],
        marker='x',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c=[0, 1, 0],
        label="C1")
ax.plot(t,xCombined[:,1],
        marker='+',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c=[0, 1, 0],
        label="C2")
ax.plot(t,xCombined[:,2],
        marker='o',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c=[0, 1, 0],
        label="C3")

ax.plot(t,x1,
        marker='x',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c = [1, 0 ,1],
        label="S1")
ax.plot(t,x2,
        marker='+',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c = [1, 0 ,1],
        label="S2")
ax.plot(t,x3,
        marker='o',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c = [1, 0 ,1],
        label="S3")
# broken
ax.plot(t,x1,
        marker='x',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c = [0.6, 0.6, 0.6],
        label="E1")
ax.plot(t,xExp[:,0],
        marker='+',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c = [0.6, 0.6, 0.6],
        label="E2")

ax.plot(t,xExp[:,1],
        marker='o',
        markersize=10,
        fillstyle='none',
        linestyle=':',
        c = [0.6, 0.6, 0.6],
        label="E3")

# Format Plot
fig.set_dpi(150)
fig.set_size_inches(9, 2.5)
ax.set_xlim(min(t), max(t))
ax.grid(True, alpha=0.25)
ax.legend(loc='lower right',  ncol=3)
fig.tight_layout()    
plt.show(block = True)
plt.pause(0.00001)
"""

