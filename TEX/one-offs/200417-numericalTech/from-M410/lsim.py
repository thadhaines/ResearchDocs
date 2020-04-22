def lsim(system, U, T, X0=None, interp=True):
    """
    Simulate output of a continuous-time linear system.
    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:
        * 1: (instance of `lti`)
        * 2: (num, den)
        * 3: (zeros, poles, gain)
        * 4: (A, B, C, D)
    U : array_like
        An input array describing the input at each time `T`
        (interpolation is assumed between given times).  If there are
        multiple inputs, then each column of the rank-2 array
        represents an input.  If U = 0 or None, a zero input is used.
    T : array_like
        The time steps at which the input is defined and at which the
        output is desired.  Must be nonnegative, increasing, and equally spaced.
    X0 : array_like, optional
        The initial conditions on the state vector (zero by default).
    interp : bool, optional
        Whether to use linear (True, the default) or zero-order-hold (False)
        interpolation for the input array.
    Returns
    -------
    T : 1D ndarray
        Time values for the output.
    yout : 1D ndarray
        System response.
    xout : ndarray
        Time evolution of the state vector.
    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).
    Examples
    --------
    Simulate a double integrator y'' = u, with a constant input u = 1
    >>> from scipy import signal
    >>> system = signal.lti([[0., 1.], [0., 0.]], [[0.], [1.]], [[1., 0.]], 0.)
    >>> t = np.linspace(0, 5)
    >>> u = np.ones_like(t)
    >>> tout, y, x = signal.lsim(system, u, t)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, y)
    """
    # ensure sys is state space
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('lsim can only be used with continuous-time '
                             'systems.')
    else:
        sys = lti(*system)._as_ss()
    T = atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError("T must be a rank-1 array.")

    # map to ABCD arrays, recieve number of states and inputs
    A, B, C, D = map(np.asarray, (sys.A, sys.B, sys.C, sys.D))
    n_states = A.shape[0]
    n_inputs = B.shape[1]

    # interperet number of steps
    n_steps = T.size

    # set states to zero if no initial states given
    if X0 is None:
        X0 = zeros(n_states, sys.A.dtype)
        
    # allocate state output array
    xout = zeros((n_steps, n_states), sys.A.dtype)

    if T[0] == 0:
        xout[0] = X0
    elif T[0] > 0:
        # step forward to initial time, with zero input
        xout[0] = dot(X0, linalg.expm(transpose(A) * T[0]))
    else:
        raise ValueError("Initial time must be nonnegative")

    no_input = (U is None or
                (isinstance(U, (int, float)) and U == 0.) or
                not np.any(U))

    if n_steps == 1:
        yout = squeeze(dot(xout, transpose(C)))
        if not no_input:
            yout += squeeze(dot(U, transpose(D)))
        return T, squeeze(yout), squeeze(xout)

    # check for consistent time step
    dt = T[1] - T[0]
    if not np.allclose((T[1:] - T[:-1]) / dt, 1.0):
        warnings.warn("Non-uniform timesteps are deprecated. Results may be "
                      "slow and/or inaccurate.", DeprecationWarning)
        return lsim2(system, U, T, X0)

    if no_input:
        # Zero input: just use matrix exponential
        # take transpose because state is a row vector
        expAT_dt = linalg.expm(transpose(A) * dt)
        for i in xrange(1, n_steps):
            xout[i] = dot(xout[i-1], expAT_dt)
        yout = squeeze(dot(xout, transpose(C)))
        return T, squeeze(yout), squeeze(xout)

    # Nonzero input
    U = atleast_1d(U)
    if U.ndim == 1:
        U = U[:, np.newaxis]

    if U.shape[0] != n_steps:
        raise ValueError("U must have the same number of rows "
                         "as elements in T.")

    if U.shape[1] != n_inputs:
        raise ValueError("System does not define that many inputs.")

    if not interp:
        # Zero-order hold
        # Algorithm: to integrate from time 0 to time dt, we solve
        #   xdot = A x + B u,  x(0) = x0
        #   udot = 0,          u(0) = u0.
        #
        # Solution is
        #   [ x(dt) ]       [ A*dt   B*dt ] [ x0 ]
        #   [ u(dt) ] = exp [  0     0    ] [ u0 ]
        M = np.vstack([np.hstack([A * dt, B * dt]),
                       np.zeros((n_inputs, n_states + n_inputs))])
        # transpose everything because the state and input are row vectors
        expMT = linalg.expm(transpose(M))
        Ad = expMT[:n_states, :n_states]
        Bd = expMT[n_states:, :n_states]
        for i in xrange(1, n_steps):
            xout[i] = dot(xout[i-1], Ad) + dot(U[i-1], Bd)
    else:
        # Linear interpolation between steps
        # Algorithm: to integrate from time 0 to time dt, with linear
        # interpolation between inputs u(0) = u0 and u(dt) = u1, we solve
        #   xdot = A x + B u,        x(0) = x0
        #   udot = (u1 - u0) / dt,   u(0) = u0.
        #
        # Solution is
        #   [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
        #   [ u(dt) ] = exp [  0     0    I ] [  u0   ]
        #   [u1 - u0]       [  0     0    0 ] [u1 - u0]
        M = np.vstack([np.hstack([A * dt, B * dt,
                                  np.zeros((n_states, n_inputs))]),
                       np.hstack([np.zeros((n_inputs, n_states + n_inputs)),
                                  np.identity(n_inputs)]),
                       np.zeros((n_inputs, n_states + 2 * n_inputs))])
        expMT = linalg.expm(transpose(M))
        Ad = expMT[:n_states, :n_states]
        Bd1 = expMT[n_states+n_inputs:, :n_states]
        Bd0 = expMT[n_states:n_states + n_inputs, :n_states] - Bd1
        for i in xrange(1, n_steps):
            xout[i] = (dot(xout[i-1], Ad) + dot(U[i-1], Bd0) + dot(U[i], Bd1))

    yout = (squeeze(dot(xout, transpose(C))) + squeeze(dot(U, transpose(D))))
    return T, squeeze(yout), squeeze(xout)

