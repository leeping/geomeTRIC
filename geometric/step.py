"""
step.py: Algorithms and tools for taking optimization steps

Copyright 2016-2020 Regents of the University of California and the Authors

Authors: Lee-Ping Wang, Chenchen Song

Contributors: 

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import division
import numpy as np
from numpy.linalg import multi_dot

from .nifty import row, col, flat, invert_svd, bohr2ang, ang2bohr, logger, pvec1d, pmat2d
from .rotate import get_rot, sorted_eigh

def between(s, a, b):
    if a < b:
        return s > a and s < b
    elif a > b:
        return s > b and s < a
    else:
        raise RuntimeError('a and b must be different')

def brent_wiki(f, a, b, rel, cvg=0.1, obj=None, verbose=0):
    """
    Brent's method for finding the root of a function.

    Parameters
    ----------
    f : function
        The function containing the root to be found
    a : float
        One side of the "bracket" to start finding the root
    b : float
        The other side of the "bracket"
    rel : float
        The denominator used to calculate the fractional error (in our case, the trust radius)
    cvg : float
        The convergence threshold for the relative error
    obj : object
        Object associated with the function that we may communicate with if desired
    verbose : int
        Print diagnostic messages

    Returns
    -------
    float
        The location of the root
    """
    fa = f(a)
    fb = f(b)
    if fa*fb > 0:
        raise RuntimeError('Not bracketed')
    if np.abs(fa) < np.abs(fb):
        # Swap if |f(a)| < |f(b)|
        a, b = b, a
        fa, fb = fb, fa
    # Set c to a
    c = a
    fc = fa
    mflag = True
    delta = 1e-6
    epsilon = min(0.01, 1e-2*np.abs(a-b))
    if obj is not None: obj.brentFailed = False
    while True:
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = a*fb*fc/((fa-fb)*(fa-fc))
            s += b*fa*fc/((fb-fa)*(fb-fc))
            s += c*fa*fb/((fc-fa)*(fc-fb))
        else:
            # Secant method
            s = b-fb*(b-a)/(fb-fa)
        # Evaluate conditions
        condition1 = not between(s, (3*a+b)/4, b)
        condition2 = mflag and (np.abs(s-b) >= np.abs(b-c)/2)
        condition3 = (not mflag) and (np.abs(s-b) >= np.abs(c-d)/2)
        condition4 = mflag and (np.abs(b-c) < delta)
        condition5 = (not mflag) and (np.abs(c-d) < delta)
        if any([condition1, condition2, condition3, condition4, condition5]):
            # Bisection method
            s = (a+b)/2
            mflag = True
        else:
            mflag = False
        # Calculate f(s)
        fs = f(s)
        # print a, s, b, fs, rel, cvg
        # Successful convergence
        if np.abs(fs/rel) <= cvg:
            return s
        # Convergence failure - interval becomes
        # smaller than threshold
        if np.abs(b-a) < epsilon:
            if verbose: logger.info("returning because interval is too small\n")
            if obj is not None: obj.brentFailed = True
            return s
        # Exit before converging when
        # the function value is positive
        if hasattr(obj, 'from_above'):
            if (obj is not None and obj.from_above) and fs > 0:
                return s
        d = c; fd = fc
        c = b; fc = fb
        if fa*fs < 0:
            b = s; fb = fs
        else:
            a = s; fa = fs
        if np.abs(fa) < np.abs(fb):
            # Swap if |f(a)| < |f(b)|
            a, b = b, a
            fa, fb = fb, fa

class Froot(object):
    """
    Object describing a function of the internal coordinate step
    length, which returns the Cartesian coordinate step length minus
    the trust radius.

    This is an object instead of a function mainly because we want the
    Brent root-finding method to read and write extra attributes of
    this function and not just its value, for example: Did we converge
    to a root? Under what conditions are we allowed to exit the algorithm?
    """
    def __init__(self, trust, v0, X, G, H, IC, params):
        self.counter = 0
        self.stores = {}
        self.trust = trust
        self.target = trust
        self.above_flag = False
        self.stored_arg = None
        self.stored_val = None
        self.brentFailed = False
        self.params = params
        self.v0 = v0
        self.X = X
        self.G = G
        self.H = H
        self.IC = IC

    def evaluate(self, trial):
        """
        This is a one-argument "function" that is called by brent_wiki which takes
        an internal coordinate step length as input, and returns the Cartesian coordinate
        step length (minus the target) as output.
        """
        v0 = self.v0
        X = self.X
        G = self.G
        H = self.H
        IC = self.IC
        trust = self.trust
        if trial == 0.0:
            self.from_above = False
            return -trust
        else:
            if trial in self.stores:
                cnorm = self.stores[trial]
                self.from_above = False
            else:
                dy, expect = trust_step(trial, v0, X, G, H, IC, self.params.transition, self.params.verbose)
                cnorm = get_cartesian_norm(X, dy, IC, self.params.enforce, self.params.verbose)
                # Early "convergence"; this signals whether we have found a valid step that is
                # above the current target, but below the original trust radius. This happens
                # when the original trust radius fails, and we reduce the target step-length
                # as a contingency
                self.from_above = (self.above_flag and not IC.bork and cnorm < trust)
                self.stores[trial] = cnorm
                self.counter += 1
            # Store the largest trial value with cnorm below the target
            if cnorm-self.target < 0:
                if self.stored_val is None or cnorm > self.stored_val:
                    self.stored_arg = trial
                    self.stored_val = cnorm
            if self.params.verbose: logger.info("  Brent Iter: %i Internal-step: %.4f Cartesian-step: %.4f ---> Trust-Radius: %.4f%s\n" % (self.counter, trial, cnorm, self.target, " (done)" if self.from_above else ""))
            return cnorm-self.target

def calc_drms_dmax(Xnew, Xold, align=True):
    """
    Align and calculate the RMSD for two geometries.

    Xnew : np.ndarray
        First set of coordinates as a flat array in a.u.
    Xold : np.ndarray
        Second set of coordinates as a flat array in a.u.
    align : bool
        Align before calculating RMSD or no?

    Returns
    -------
    float, float
        RMS and maximum displacements in Angstrom
    """
    # Shift to the origin
    Xold = Xold.copy().reshape(-1, 3)
    Xold -= np.mean(Xold, axis=0)
    Xnew = Xnew.copy().reshape(-1, 3)
    Xnew -= np.mean(Xnew, axis=0)
    # Obtain the rotation
    if align:
        U = get_rot(Xnew, Xold)
        # Xrot = np.array((U*np.matrix(Xnew).T).T).flatten()
        Xrot = np.dot(U, Xnew.T).T.flatten()
        Xold = np.array(Xold).flatten()
        displacement = np.sqrt(np.sum((((Xrot-Xold)*bohr2ang).reshape(-1,3))**2, axis=1))
    else:
        displacement = np.sqrt(np.sum((((Xnew-Xold)*bohr2ang).reshape(-1,3))**2, axis=1))
    rms_displacement = np.sqrt(np.mean(displacement**2))
    max_displacement = np.max(displacement)
    return rms_displacement, max_displacement

def get_cartesian_norm(X, dy, IC, enforce=0.0, verbose=0):
    """
    Get the norm of the optimization step in Cartesian coordinates.

    Parameters
    ----------
    X : np.ndarray
        Nx3 array of Cartesian coordinates in atomic units
    dy : np.ndarray
        N_ic array of internal coordinate displacements
    IC : InternalCoordinates
        Object describing the internal coordinate system
    enforce : float
        Enforce constraints in the internal coordinate system when
        all constraints are satisfied to within the provided tolerance.
        Passing a value of zero means this is not used.
    verbose : int
        Print diagnostic messages

    Returns
    -------
    float
        The RMSD between the updated and original Cartesian coordinates
    """
    # Displacement of each atom in Angstrom
    if IC.haveConstraints() and enforce > 0.0:
        Xnew = IC.newCartesian_withConstraint(X, dy, thre=enforce, verbose=verbose)
    else:
        Xnew = IC.newCartesian(X, dy, verbose=verbose)
    rmsd, maxd = calc_drms_dmax(Xnew, X)
    return rmsd

def rebuild_hessian(IC, H0, coord_seq, grad_seq, params):
    """
    Rebuild the Hessian after making a change to the internal coordinate system.

    Parameters
    ----------
    IC : InternalCoordinates
        Object describing the internal coordinate system
    H0 : np.ndarray
        N_ic x N_ic square matrix containing the guess Hessian
    coord_seq : list
        List of N_atom x 3 Cartesian coordinates in atomic units
    grad_seq : list
        List of N_atom x 3 Cartesian gradients in atomic units
    params : OptParams object
        Uses trust, epsilon, and reset
        trust : Only recover using previous geometries within the trust radius
        epsilon : Small eigenvalue threshold
        reset : Revert to the guess Hessian if eigenvalues smaller than threshold

    Returns
    -------
    np.ndarray
        Internal coordinate Hessian updated with series of internal coordinate gradients
    """
    Na = len(coord_seq[0])/3
    history = 0
    for i in range(2, len(coord_seq)+1):
        disp = bohr2ang*(coord_seq[-i]-coord_seq[-1])
        rmsd = np.sqrt(np.sum(disp**2)/Na)
        if rmsd > params.trust: break
        history += 1
    if history < 1:
        return H0.copy()
    logger.info("Rebuilding Hessian using %i gradients\n" % history)
    y_seq = [IC.calculate(i) for i in coord_seq[-history-1:]]
    g_seq = [IC.calcGrad(i, j) for i, j in zip(coord_seq[-history-1:],grad_seq[-history-1:])]
    Yprev = y_seq[0]
    Gprev = g_seq[0]
    H = H0.copy()
    for i in range(1, len(y_seq)):
        Y = y_seq[i]
        G = g_seq[i]
        Yprev = y_seq[i-1]
        Gprev = g_seq[i-1]
        Dy   = col(Y - Yprev)
        Dg   = col(G - Gprev)
        # Mat1 = (Dg*Dg.T)/(Dg.T*Dy)[0,0]
        # Mat2 = ((H*Dy)*(H*Dy).T)/(Dy.T*H*Dy)[0,0]
        Mat1 = np.dot(Dg,Dg.T)/np.dot(Dg.T,Dy)[0,0]
        Mat2 = np.dot(np.dot(H,Dy),np.dot(H,Dy).T)/multi_dot([Dy.T,H,Dy])[0,0]
        Hstor = H.copy()
        H += Mat1-Mat2
    if np.min(np.linalg.eigh(H)[0]) < params.epsilon and params.reset:
        logger.info("Eigenvalues below %.4e (%.4e) - returning guess\n" % (params.epsilon, np.min(np.linalg.eigh(H)[0])))
        return H0.copy()
    return H

def image_gradient_hessian(G, H, indices):
    """
    Calculate an image quadratic function 
    """
    # Sorted eigenvalues and corresponding eigenvectors of the Hessian
    Hvals, Hvecs = sorted_eigh(H, asc=True)

    # Projection of gradient along the Hessian eigenvectors
    # Gproj = np.dot(Hvecs.T, G)

    house = np.eye(G.shape[0])
    
    for i in indices:
        Hvals[i] *= -1
        house -= 2*np.outer(Hvecs[:,i], Hvecs[:,i])

    Gs = np.dot(house, G)
        
    Hs = np.zeros_like(H)
    # Gs = np.zeros_like(G)
    for i in range(H.shape[0]):
        Hs += Hvals[i] * np.outer(Hvecs[:,i], Hvecs[:,i])
        # Gs += Gproj[i] * Hvecs[:,i]
    
    return Gs, Hs

def get_delta_prime_trm(v, X, G, H, IC, verbose=0):
    """
    Returns the Newton-Raphson step given a multiple of the diagonal
    added to the Hessian, the expected decrease in the energy, and
    the derivative of the step length w/r.t. v.

    Parameters
    ----------
    v : float
        Number that is added to the Hessian diagonal
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    # LPW 2020-01-24 Testing whether the image potential can be combined with NR for transition state optimization
    # Gs, Hs = image_gradient_hessian(G, H, [0])
    if IC is not None:
        GC, HC = IC.augmentGH(X, G, H) if IC.haveConstraints() else (G, H)
    else:
        GC, HC = (G, H)
    HT = HC + v*np.eye(len(HC))
    # The constrained degrees of freedom should not have anything added to diagonal
    for i in range(len(G), len(GC)):
        HT[i, i] = 0.0
    if verbose >= 2:
        seig = sorted(np.linalg.eig(HT)[0])
        logger.info("sorted(eig) : % .5e % .5e % .5e ... % .5e % .5e % .5e\n" % (seig[0], seig[1], seig[2], seig[-3], seig[-2], seig[-1]))
    try:
        Hi = invert_svd(HT)
    except:
        logger.info("\x1b[1;91mSVD Error - increasing v by 0.001 and trying again\x1b[0m\n")
        return get_delta_prime_trm(v+0.001, X, G, H, IC)
    dyc = flat(-1 * np.dot(Hi,col(GC)))
    dy = dyc[:len(G)]

    d_prime = flat(-1 * np.dot(Hi, col(dyc)))[:len(G)]
    dy_prime = np.dot(dy,d_prime)/np.linalg.norm(dy)
    # sol = flat(0.5*row(dy)*np.matrix(H)*col(dy))[0] + np.dot(dy,G)
    sol = flat(0.5*multi_dot([row(dy),H,col(dy)]))[0] + np.dot(dy,G)
    return dy, sol, dy_prime

def rfo_gen_evp(M, a):
    """
    Solve the generalized eigenvalue problem often encountered
    in restricted step RFO, given by:

    M * v = l * A * v

    where M is any matrix (always symmetric in our case),
    v and l are the desired eigenvectors,
    and A is a scaling matrix with the form:

    [[ 1  0  ]
     [ 0 a*I ]]
    """
    # Values on the diagonal of the matrix
    av = np.ones(M.shape[0])
    av[1:] = a
    # A to the -1/2 power
    Amh = np.diag(av**-0.5)
    # Form A^-1/2 * M * A^-1/2
    Mp = multi_dot([Amh, M, Amh])
    # Solve for eigenvalues and eigenvectors, which are actually v' = (A^1/2)*v
    eigvals, eigvecsp = sorted_eigh(Mp, asc=True)
    # Get the eigenvectors of the generalized EVP as v = (A^-1/2) v'
    eigvecs = np.dot(Amh, eigvecsp)
    return eigvals, eigvecs

def get_delta_prime_rs_p_rfo(alpha, X, G, H, IC, verbose=0):
    """
    Return the restricted-step rational functional optimization
    step, given a particular value of alpha. The step is given by:
    1) Solving the generalized eigenvalue problem
    [[0 G]  = lambda * [[1 0] * vec ,
     [G H]]             [0 S]]
       where the LHS matrix is called the augmented Hessian,
       and S is alpha times the identity (starting value 1.0).
    2) Dividing vec through by the 0th element, and keeping the rest
    This function also calculates the derivative of the norm of the step
    with respect to alpha, which allows trust_step() to rapidly find
    the RS-RFO step that satisfies a desired step length.

    Currently does not work with constraints, and gives equivalent performance
    to the trust radius method.

    Parameters
    ----------
    alpha : float
        Multiple of the identity in the S-matrix
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    if IC.haveConstraints():
        raise RuntimeError("Rational function optimization does not support constraints")
    # verbose = 2
    if verbose >= 4:
        logger.info("        === RS-P-RFO method with alpha = %.5f ===\n" % alpha)

    # Sorted eigenvalues and corresponding eigenvectors of the Hessian
    Hvals, Hvecs = sorted_eigh(H, asc=True)

    # Projection of gradient along the Hessian eigenvectors
    Gproj = np.dot(Hvecs.T, G)

    # Indices of the transition vector (which forms the small RFO partition) and the other vectors
    tv = 0
    ot = np.array([i for i in range(H.shape[0]) if i != tv])

    # The P-RFO matrices follow the convention of Bofill (1998)
    # with the "extra" row in the upper left.
    # Form the small P-RFO matrix, which is a 2x2 matrix
    prfo_tv = np.array([[0, Gproj[tv]], [Gproj[tv], Hvals[tv]]])
    # Form the large P-RFO matrix, which is the same size as H itself
    prfo_ot = np.zeros_like(H)
    # Fill matrix (except top left) with the non-'tv' eigenvalues
    prfo_ot[1:, 1:] = np.diag(Hvals[ot])
    # Fill the top row and left column with the non-'tv' gradient components
    prfo_ot[0, 1:] = Gproj[ot]
    prfo_ot[1:, 0] = Gproj[ot]

    # Solve the generalized eigenvector problems
    tv_vals, tv_vecs = rfo_gen_evp(prfo_tv, alpha)
    ot_vals, ot_vecs = rfo_gen_evp(prfo_ot, alpha)

    ## Form the P-RFO step.
    dy_tv = -Gproj[tv]*Hvecs[:,tv]/(Hvals[tv]-tv_vals[-1]*alpha)
    denoms = np.zeros_like(dy_tv)
    dy_coeffs = np.zeros_like(dy_tv)
    dy_coeffs[tv] = -Gproj[tv]/(Hvals[tv]-tv_vals[-1]*alpha)
    denoms[tv] = Hvals[tv]-tv_vals[-1]*alpha
    # Compute transition vector contribution to |dy|^2/d(alpha)
    dy2_prime_tv = 2 * (tv_vals[-1]/(1+alpha*np.dot(dy_tv,dy_tv))) * (Gproj[tv]**2/(Hvals[tv]-tv_vals[-1]*alpha)**3)
    # Now for the other vectors
    dy_ot = np.zeros_like(dy_tv)
    dy2_prime_ot = 0.0
    for i in ot:
        denom = Hvals[i]-ot_vals[0]*alpha
        denoms[i] = denom
        # Numerical instabilities can occur when the shift is very close to a pole,
        # so we exclude these components.
        if np.abs(denom) > 1e-5:
            dy_ot -= Gproj[i]*Hvecs[:,i]/denom
            dy_coeffs[i] = -Gproj[i]/denom
            dy2_prime_ot += Gproj[i]**2/denom**3
    dy2_prime_ot *= 2 * (ot_vals[0]/(1+alpha*np.dot(dy_ot,dy_ot)))
    # Add the transition vector and other vector contributions together
    dy = dy_tv + dy_ot
    dy2_prime = dy2_prime_tv + dy2_prime_ot
    # Derivative of the norm of the step w/r.t. alpha
    dy_prime = dy2_prime/(2*np.linalg.norm(dy))
    # For some reason, this formula from the paper suffers from some numerical problems
    # expect = (tv_vals[-1]/tv_vecs[0,-1]**2 + ot_vals[0]/ot_vecs[0,0]**2) / 2
    # Use the quadratic approximation to get expected change in the energy
    expect = flat(0.5*multi_dot([row(dy),H,col(dy)]))[0] + np.dot(dy,G)

    if verbose >= 5:
        logger.info("        Largest / smallest eigvals of small / large P-RFO matrix: % .5f % .5f\n" % (tv_vals[-1], ot_vals[0]))
        logger.info("        Small P-RFO matrix:\n        ")
        pmat2d(prfo_tv, precision=5, format='f')
        logger.info("        Eigenvalues of small P-RFO matrix:\n        ")
        pvec1d(tv_vals, precision=5, format='f')
        logger.info("        Large P-RFO matrix:\n        ")
        pmat2d(prfo_ot, precision=5, format='f')
        logger.info("        Eigenvalues of large P-RFO matrix:\n        ")
        pvec1d(ot_vals, precision=5, format='f')
        logger.info("        l_max*alpha, l_min*alpha = %.5f %.5f\n" % (tv_vals[-1]*alpha, ot_vals[0]*alpha))
        logger.info("        Numerator   of Gproj[i]/(h[i]-l*a) along each mode:\n        ")
        pvec1d(Gproj)
        logger.info("        Denominator of Gproj[i]/(h[i]-l*a) along each mode:\n        ")
        pvec1d(denoms)
        logger.info("        Coefficients of P-RFO step along each mode:\n        ")
        pvec1d(dy_coeffs)
        logger.info("        Step obtained from P-RFO method:\n        ")
        pvec1d(dy)
    elif verbose >= 4:
        logger.info("        Largest / smallest eigvals of small / large P-RFO matrix: % .5f % .5f\n" % (tv_vals[-1], ot_vals[0]))
        logger.info("        l_max*alpha(TS), l_min*alpha(min) = %.5f %.5f\n" % (tv_vals[-1]*alpha, ot_vals[0]*alpha))
        logger.info("        Coefficients of P-RFO step along normal modes:\n")
        printIdxs = list(np.argsort(np.abs(dy_coeffs))[-4:])
        if 0 not in printIdxs: printIdxs.append(0)
        for i in sorted(printIdxs):
            logger.info("          dy[%3i] = % .6f\n" % (i, dy_coeffs[i]))
        # pvec1d(dy_coeffs, precision=3, format='f')

    return dy, expect, dy_prime

def get_delta_prime_rfo(alpha, X, G, H, IC, verbose=0):
    """
    Return the restricted-step rational functional optimization
    step, given a particular value of alpha. The step is given by:
    1) Solving the generalized eigenvalue problem
    [[0 G]  = lambda * [[1 0] * vec ,
     [G H]]             [0 S]]
       where the LHS matrix is called the augmented Hessian,
       and S is alpha times the identity (starting value 1.0).
    2) Dividing vec through by the 0th element, and keeping the rest
    This function also calculates the derivative of the norm of the step
    with respect to alpha, which allows trust_step() to rapidly find
    the RS-RFO step that satisfies a desired step length.

    Currently does not work with constraints, and gives equivalent performance
    to the trust radius method.

    Parameters
    ----------
    alpha : float
        Multiple of the identity in the S-matrix
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    try:
        import scipy
    except ImportError:
        raise ImportError("RFO optimization requires scipy package. If this becomes important in the future, scipy will become a required dependency.")
    if IC.haveConstraints():
        raise RuntimeError("Still need to implement RFO with constraints")
    S = alpha*np.eye(len(H))
    # Augmented Hessian matrix
    AH = np.zeros((H.shape[0]+1, H.shape[1]+1), dtype=float)
    AH[1:, 1:] = H
    AH[0, 1:] = G
    AH[1:, 0] = G
    B = np.zeros_like(AH)
    B[0,0] = 1.0
    B[1:,1:] = S
    # Solve the generalized eigenvalue problem
    AHeig, AHvec = scipy.linalg.eigh(AH, b=B)
    lmin = AHeig[0]
    # logger.info("AH eigenvalues: %.5e %.5e %.5e ... %.5e %.5e %.5e" % (AHeig[0],AHeig[1],AHeig[2],AHeig[-3],AHeig[-2],AHeig[-1]))
    vmin = np.array(AHvec[:, 0]).flatten()
    dy = (vmin / vmin[0])[1:]
    nu = alpha*lmin
    # Now get eigenvectors of the Hessian
    Heig, Hvec = sorted_eigh(H, asc=True)
    Hvec = np.array(Hvec)
    dyprime2 = 0
    dy2 = 0
    for i in range(H.shape[0]):
        dyprime2 += np.dot(Hvec[:,i].T,G)**2/(Heig[i]-nu)**3
        dy2 += np.dot(Hvec[:,i].T,G)**2/(Heig[i]-nu)**2
    dyprime2 *= (2*lmin)/(1+alpha*np.dot(dy,dy))
    expect = lmin/2*(1+multi_dot([row(dy),S,col(dy)]))[0]
    dyprime1 = dyprime2 / (2*np.sqrt(dy2))
    return dy, expect, dyprime1

def get_delta_prime(v, X, G, H, IC, rfo, verbose=0):
    """
    Return the internal coordinate step given a parameter "v".
    "v" refers to the multiple of the identity added to the Hessian
    in trust-radius Newton Raphson (TRM), and the multiple of the
    identity on the RHS matrix in rational function optimization (RFO).
    Note that reasonable default values are v = 0.0 in TRM and 1.0 in RFO.

    Parameters
    ----------
    v : float
        Number that is added to the Hessian diagonal
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    rfo : bool
        If True, use rational functional optimization, otherwise use trust-radius method
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step
    expect : float
        Expected change of the objective function
    dy_prime : float
        Derivative of the internal coordinate step size w/r.t. v
    """
    if rfo:
        method="Restricted step P-RFO"
        dy, expect, dy_prime = get_delta_prime_rs_p_rfo(v, X, G, H, IC, verbose)
    else:
        method="Trust radius Newton-Raphson"
        dy, expect, dy_prime = get_delta_prime_trm(v, X, G, H, IC, verbose)
    if verbose >= 2: logger.info("      %s step with v = %10.5f : expect-deltaE = %.5f, Internal-step = %.5f\n" % (method, v, expect, np.linalg.norm(dy)))
    return dy, expect, dy_prime

def trust_step(target, v0, X, G, H, IC, rfo, verbose=0):
    """
    Apply an iteration formula to find the trust radius step,
    given the target value of the trust radius.

    Parameters
    ----------
    target : float
        Target size of the trust radius step
    v0 : float
        Initial guess for Number that is added to the Hessian diagonal
    X : np.ndarray
        Flat array of Cartesian coordinates in atomic units
    G : np.ndarray
        Flat array containing internal gradient
    H : np.ndarray
        Square array containing internal Hessian
    IC : InternalCoordinates
        Object describing the internal coordinate system
    rfo : bool
        If True, use rational functional optimization, otherwise use trust-radius method
    verbose : int
        Print diagnostic messages

    Returns
    -------
    dy : np.ndarray
        The internal coordinate step with the desired size
    sol : float
        Expected change of the objective function
    """
    if verbose >= 2: logger.info("    trust_step targeting internal coordinate step of length %.4f\n" % target)
    dy, sol, dy_prime = get_delta_prime(v0, X, G, H, IC, rfo, verbose)

    ndy = np.linalg.norm(dy)
    if ndy < target:
        return dy, sol
    v = v0
    niter = 0
    ndy_last = 0
    # Store the minimum norm in case we give up
    m_ndy = ndy
    m_dy = dy.copy()
    m_sol = sol
    while True:
        if rfo:
            # Nonlinear Newton's method solution, from Bofill (1998)
            v += (target/ndy-1)*(ndy/dy_prime) 
        else:
            # Iterative formula from Hebden (1973), equation 5.2.10 in "Practical methods of optimization" by Fletcher
            v += (1-ndy/target)*(ndy/dy_prime)
        dy, sol, dy_prime, = get_delta_prime(v, X, G, H, IC, rfo, verbose)
        ndy = np.linalg.norm(dy)
        if np.abs((ndy-target)/target) < 0.001:
            if verbose >= 3: get_delta_prime(v, X, G, H, IC, rfo, verbose+1)
            if verbose: logger.info("    trust_step Iter:  %4i, v = %.5f, dy on target:   %.5f ---> %.5f\n" % (niter, v, ndy, target))
            return dy, sol
        # With Lagrange multipliers it may be impossible to go under a target step size
        elif niter > 10 and np.abs(ndy_last-ndy)/ndy < 0.001:
            if verbose >= 3: get_delta_prime(v, X, G, H, IC, rfo, verbose+1)
            if verbose: logger.info("    trust_step Iter:  %4i, v = %.5f, dy over target: %.5f -x-> %.5f\n" % (niter, v, ndy, target))
            return dy, sol
        elif verbose >= 2:
            logger.info("    trust_step Iter:  %4i, v = %.5f, dy -> target:   %.5f ---> %.5f\n" % (niter, v, ndy, target))
        niter += 1
        ndy_last = ndy
        if ndy < m_ndy:
            m_ndy = ndy
            m_dy = dy.copy()
            m_sol = sol
        # Break out of infinite oscillation loops
        if niter%100 == 99:
            logger.info("    trust_step Iter:  %4i, randomizing\n" % niter)
            v += np.random.random() * niter / 100
        if niter%1000 == 999:
            if verbose >= 3: get_delta_prime(v, X, G, H, IC, rfo, verbose+1)
            if verbose: logger.info("    trust_step Iter:  %4i, v = %.5f, dy at max-iter: %.5f -x-> %.5f\n" % (niter, v, ndy, target))
            return m_dy, m_sol
