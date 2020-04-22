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

# Method Definitions
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
for caseN in range(0,2):#range(0,3):

    if caseN == 0:
        # step input Integrator example
        caseName = 'Step Input Integrator Example'
        tStart =0
        tEnd = 4
        numPoints = 4
        blkFlag = False # for holding plots open

        U = 1
        ic = [0,0] # initial condition x,y
        fp = lambda x, y: 1
        f = lambda x, c: x+c
        findC = lambda x, y: y-x

        system = signal.lti([1],[1,0])

        calcInt = 0.5*(tEnd**2) # Calculated integral

    else:
        # step input Low pass example
        caseName = 'Step Input Low Pass Example'
        tStart =0
        tEnd = 2
        numPoints = 4
        blkFlag = True # for holding plots open

        A = 0.25
        U = 1.0
        ic = [0,0] # initial condition x,y
        fp = lambda x, y: 1/A*np.exp(-x/A)# via table
        f = lambda x, c: -np.exp(-x/A) +c
        findC = lambda x, y : y+np.exp(-x/A)

        system = signal.lti([1],[A,1])

        calcInt = tEnd + A*np.exp(-tEnd/A)-A # Calculated integral

    # Find C from integrated equation for exact soln
    c = findC(ic[0], ic[1])

    # Initialize current value dictionary
    # Shown to mimic PSLTDSim record keeping
    cv={
        't' :ic[0],
        'yRK': ic[1], 
        'ySI': ic[1],
        'yLS': ic[1],
        }

    # Calculate time step
    ts = (tEnd-tStart)/numPoints

    # Calculate exact solution
    tExact = np.linspace(tStart,tEnd, 1000)
    yExact = f(tExact, c)

    # Initialize running value lists
    t=[]
    yRK =[]
    # solve ivp
    ySI = []
    tSI = []
    # lsim
    yLS = []
    xLS = [] # required to track state history

    t.append(cv['t'])
    yRK.append(cv['yRK'])
    yLS.append(cv['yLS'])
    xLS.append(cv['yLS'])    

    # Start Simulation
    while cv['t']< tEnd:

        # Calculate Runge-Kutta result
        cv['yRK'] = rk45( fp, cv['t'], cv['yRK'],  ts )

        # Runge-Kutta 4(5) via solve IVP.
        soln = solve_ivp(fp, (cv['t'], cv['t']+ts), [cv['ySI']])

        # lsim solution
        tout, ylsim, xlsim = signal.lsim(system, [U,U], [0,ts], xLS[-1])            
        # Log calculated results
        yRK.append(cv['yRK'])
        
        # handle solve_ivp output data
        ySI += list(soln.y[-1])
        tSI += list(soln.t)
        cv['ySI'] = ySI[-1] # ensure correct cv

        # handle lsim output data
        cv['yLS']=ylsim[-1]
        yLS.append(cv['yLS'])
        xLS.append(xlsim[-1]) # this is the state
        
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
    ax.plot(t,yRK,
            marker='*',
            markersize=10,
            fillstyle='none',
            linestyle=':',
            c=[1,0,1],
            label="RK45")
    ax.plot(tSI,ySI,
            marker='x',
            markersize=10,
            fillstyle='none',
            linestyle=':',
            c=[1,.647,0],
            label="solve_ivp") 
    ax.plot(t,yLS,
            marker='+',
            markersize=10,
            fillstyle='none',
            linestyle=':',
            c ="#17becf",
            label="lsim")

    # Format Plot
    fig.set_dpi(150)
    fig.set_size_inches(9, 2.5)
    ax.set_xlim(min(t), max(t))
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best',  ncol=2)
    fig.tight_layout()    
    plt.show(block = blkFlag)
    plt.pause(0.00001)

    # Trapezoidal Integration
    exactI = trapezoidalPost(tExact,yExact)
    SIint = trapezoidalPost(tSI,ySI)
    RKint = trapezoidalPost(t,yRK)
    LSint = trapezoidalPost(t,yLS)

    print("\nMethod: Trapezoidal Int\t Absolute Error from calculated")
    print("Exact: \t%.9f\t%.9f" % (exactI ,abs(calcInt-exactI)))    
    print("RK4: \t%.9f\t%.9f" % (RKint,abs(calcInt-RKint)))
    print("SI: \t%.9f\t%.9f" % (SIint,abs(calcInt-SIint)))
    print("lsim: \t%.9f\t%.9f" % (LSint,abs(calcInt-LSint)))




