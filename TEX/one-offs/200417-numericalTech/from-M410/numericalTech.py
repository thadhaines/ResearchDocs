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
    

# Case Selection
for caseN in range(0,3):

    if caseN == 0:
        # Trig example
        caseName = 'Sinusodial Example'
        tStart =0
        tEnd = 1.5
        numPoints = 6
        blkFlag = False # for holding plots open

        ic = [0,0] # initial condition x,y
        fp = lambda x, y: -2*np.pi*np.cos(2*np.pi*x)
        f = lambda x,c: -np.sin(2*np.pi*x)+c
        findC = lambda x,y: y+2*np.pi*np.sin(2*np.pi*x)

        calcInt = 1/(2*np.pi)*np.cos(2*np.pi*1.5)-1/(2*np.pi)

    elif caseN ==1:
        # Exp example
        caseName = 'Exponential Example'
        tStart =0
        tEnd = 2
        numPoints = 4
        blkFlag = False # for holding plots open

        ic = [0,0] # initial condition x,y
        fp = lambda x, y: np.exp(x)
        f = lambda x,c: np.exp(x)+c
        findC = lambda x, y: y-np.exp(x)

        calcInt = np.exp(2)-3 # when A = 1

    else:
        # Log example
        caseName = 'Logarithmic Example'
        tStart =1
        tEnd = 3
        numPoints = 4
        blkFlag = True # for holding plots open

        ic = [1,1] # initial condition x,y
        fp = lambda x, y: 1/x
        f = lambda x,c: np.log(x)+c
        findC = lambda x, y: y-np.log(x)

        calcInt = 3*np.log(3)

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
    tExact = np.linspace(tStart,tEnd, 10000)
    yExact = f(tExact, c)

    # Initialize running value lists
    t=[]
    yE=[]
    yRK =[]
    yAB = []
    ySI = []
    tSI = []

    t.append(cv['t'])
    yE.append(cv['yE'])
    yRK.append(cv['yRK'])
    yAB.append(cv['yAB'])

    # Start Simulation
    while cv['t']< tEnd:

        # Calculate Euler result
        cv['yE'] = euler( fp, cv['t'], cv['yE'],  ts )
        # Calculate Runge-Kutta result
        cv['yRK'] = rk45( fp, cv['t'], cv['yRK'],  ts )

        # Calculate Adams-Bashforth result
        if len(t)>=2:
            cv['yAB'] = adams2( fp, cv['t'], cv['yAB'], t[-2], yAB[-2], ts )
        else:
            # Required to handle first step when a -2 index doesn't exist
            cv['yAB'] = adams2( fp, cv['t'], cv['yAB'], t[-1], yAB[-1], ts )

        # Runge-Kutta via solve IVP.
        soln = solve_ivp(fp, (cv['t'], cv['t']+ts), [cv['ySI']])
        
        # Log calculated results
        yE.append(cv['yE'])
        yRK.append(cv['yRK'])
        yAB.append(cv['yAB'])
        
        # handle solve_ivp data output
        ySI += list(soln.y[-1])
        tSI += list(soln.t)
        # ensure correct cv
        cv['ySI'] = ySI[-1]
    
        # Increment and log time
        cv['t'] += ts
        t.append(cv['t'])

    # Generate Plot
    fig, ax = plt.subplots()
    ax.set_title('Approximation Comparison\n' + caseName)
    
    #Plot all lines
    ax.plot(tExact,yExact,
            c=[0,0,0],
            linewidth=2,
            label="Exact")
    ax.plot(tSI,ySI,
            marker='x',
            markersize=10,
            fillstyle='none',
            linestyle=':',
            c=[1,.647,0],
            label="solve_ivp")    
    ax.plot(t,yE,
            marker='o',
            fillstyle='none',
            linestyle=':',
            c=[0.7,0.7,0.7],
            label="Euler")
    ax.plot(t,yRK,
            marker='*',
            markersize=10,
            fillstyle='none',
            linestyle=':',
            c=[1,0,1],
            label="RK45")
    ax.plot(t,yAB,
            marker='s',
            fillstyle='none',
            linestyle=':',
            c =[0,1,0],
            label="AB2")

    # Format Plot
    fig.set_dpi(150)
    fig.set_size_inches(9, 2.5)
    ax.set_xlim(min(t), max(t))
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best',  ncol=3)
    fig.tight_layout()    
    plt.show(block = blkFlag)
    plt.pause(0.00001)

    # Trapezoidal Integration
    exactI = trapezoidalPost(tExact,yExact)
    SIint = trapezoidalPost(tSI,ySI)
    Eint = trapezoidalPost(t,yE)
    RKint = trapezoidalPost(t,yRK)
    ABint = trapezoidalPost(t,yAB)

    print("\nMethod: Trapezoidal Int\t Absolute Error from calculated")
    print("Exact: \t%.9f\t%.9f" % (exactI ,abs(calcInt-exactI)))
    print("SI: \t%.9f\t%.9f" % (SIint,abs(calcInt-SIint)))
    print("RK4: \t%.9f\t%.9f" % (RKint,abs(calcInt-RKint)))
    print("AB2: \t%.9f\t%.9f" % (ABint,abs(calcInt-ABint)))
    print("Euler: \t%.9f\t%.9f" % (Eint,abs(calcInt-Eint)))




