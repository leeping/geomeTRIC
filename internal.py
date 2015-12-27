#!/usr/bin/env python

from __future__ import division
import numpy as np
import networkx as nx
import itertools
from copy import deepcopy
from forcebalance.nifty import click
from forcebalance.molecule import Molecule, Elements, Radii
from collections import OrderedDict, defaultdict
from scipy import optimize

class CartesianX(object):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w

    def __repr__(self):
        return "Cartesian-X %i : Weight %.3f" % (self.a+1, self.w)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return self.a == other.a

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][0]*self.w
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives[self.a][0] = self.w
        return derivatives

class CartesianY(object):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w

    def __repr__(self):
        return "Cartesian-Y %i : Weight %.3f" % (self.a+1, self.w)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return self.a == other.a

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][1]*self.w
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives[self.a][1] = self.w
        return derivatives

class CartesianZ(object):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w

    def __repr__(self):
        return "Cartesian-Z %i : Weight %.3f" % (self.a+1, self.w)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return self.a == other.a

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][2]*self.w
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives[self.a][2] = self.w
        return derivatives

class MultiCartesianX(object):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w

    def __repr__(self):
        return "MultiCartesian-X %s : Weight %.3f" % (' '.join([str(i+1) for i in self.a]), self.w)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return set(self.a) == set(other.a)

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,0])*self.w
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i in self.a:
            derivatives[i][0] = self.w
        return derivatives

class MultiCartesianY(object):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w

    def __repr__(self):
        return "MultiCartesian-Y %s : Weight %.3f" % (' '.join([str(i+1) for i in self.a]), self.w)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return set(self.a) == set(other.a)

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,1])*self.w
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i in self.a:
            derivatives[i][1] = self.w
        return derivatives

class MultiCartesianZ(object):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w

    def __repr__(self):
        return "MultiCartesian-Z %s : Weight %.3f" % (' '.join([str(i+1) for i in self.a]), self.w)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return set(self.a) == set(other.a)

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,2])*self.w
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i in self.a:
            derivatives[i][2] = self.w
        return derivatives

class Distance(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        if a == b:
            raise RuntimeError('a and b must be different')

    def __repr__(self):
        return "Distance %i-%i" % (self.a+1, self.b+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if self.b == other.b:
                return True
        if self.a == other.b:
            if self.b == other.a:
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        return np.sqrt(np.sum((xyz[a]-xyz[b])**2))
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a
        n = self.b
        u = (xyz[m] - xyz[n]) / np.linalg.norm(xyz[m] - xyz[n])
        derivatives[m, :] = u
        derivatives[n, :] = -u
        return derivatives

class Angle(object):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        if len(set([a, b, c])) != 3:
            raise RuntimeError('a, b, and c must be different')

    def __repr__(self):
        return "Angle %i-%i-%i" % (self.a+1, self.b+1, self.c+1)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.b == other.b:
            if self.a == other.a:
                if self.c == other.c:
                    return True
            if self.a == other.c:
                if self.c == other.a:
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # vector from first atom to central atom
        vector1 = xyz[a] - xyz[b]
        # vector from last atom to central atom
        vector2 = xyz[c] - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        dot = np.dot(vector1, vector2)
        return np.arccos(dot / (norm1 * norm2))

    def normal_vector(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # vector from first atom to central atom
        vector1 = xyz[a] - xyz[b]
        # vector from last atom to central atom
        vector2 = xyz[c] - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        crs = np.cross(vector1, vector2)
        crs /= np.linalg.norm(crs)
        return crs
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a
        o = self.b
        n = self.c
        # Unit displacement vectors
        u_prime = (xyz[m] - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyz[n] - xyz[o])
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm
        VECTOR1 = np.array([1, -1, 1]) / np.sqrt(3)
        VECTOR2 = np.array([-1, 1, 1]) / np.sqrt(3)
        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            # if they're parallel
            if ((np.linalg.norm(u + VECTOR1) < 1e-10) or
                    (np.linalg.norm(u - VECTOR2) < 1e-10)):
                # and they're parallel o [1, -1, 1]
                w_prime = np.cross(u, VECTOR2)
            else:
                w_prime = np.cross(u, VECTOR1)
        else:
            w_prime = np.cross(u, v)
        w = w_prime / np.linalg.norm(w_prime)
        term1 = np.cross(u, w) / u_norm
        term2 = np.cross(w, v) / v_norm
        derivatives[m, :] = term1
        derivatives[n, :] = term2
        derivatives[o, :] = -(term1 + term2)
        return derivatives

class Dihedral(object):
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        if len(set([a, b, c, d])) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        return "Dihedral Angle %i-%i-%i-%i" % (self.a+1, self.b+1, self.c+1, self.d+1)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if self.b == other.b:
                if self.c == other.c:
                    if self.d == other.d:
                        return True
        if self.a == other.d:
            if self.b == other.c:
                if self.c == other.b:
                    if self.d == other.a:
                        return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        vec1 = xyz[b] - xyz[a]
        vec2 = xyz[c] - xyz[b]
        vec3 = xyz[d] - xyz[c]
        cross1 = np.cross(vec2, vec3)
        cross2 = np.cross(vec1, vec2)
        arg1 = np.sum(np.multiply(vec1, cross1)) * \
               np.sqrt(np.sum(vec2**2))
        arg2 = np.sum(np.multiply(cross1, cross2))
        answer = np.arctan2(arg1, arg2)
        return answer
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a
        o = self.b
        p = self.c
        n = self.d
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        if (1 - np.dot(u, w)**2) < 1e-6:
            term1 = np.cross(u, w) * 0
            term3 = np.cross(u, w) * 0
        else:
            term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
            term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        if (1 - np.dot(v, w)**2) < 1e-6:
            term2 = np.cross(v, w) * 0
            term4 = np.cross(v, w) * 0
        else:
            term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
            term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        # term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        # term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        # term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        derivatives[m, :] = term1
        derivatives[n, :] = -term2
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives

class OutOfPlane(object):
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        if len(set([a, b, c, d])) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        return "Improper Dihedral Angle %i-%i-%i-%i" % (self.a+1, self.b+1, self.c+1, self.d+1)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if set([self.b, self.c, self.d]) == set([other.b, other.c, other.d]):
                return True
        #     if self.b == other.b:
        #         if self.c == other.c:
        #             if self.d == other.d:
        #                 return True
        # if self.a == other.d:
        #     if self.b == other.c:
        #         if self.c == other.b:
        #             if self.d == other.a:
        #                 return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        vec1 = xyz[b] - xyz[a]
        vec2 = xyz[c] - xyz[b]
        vec3 = xyz[d] - xyz[c]
        cross1 = np.cross(vec2, vec3)
        cross2 = np.cross(vec1, vec2)
        arg1 = np.sum(np.multiply(vec1, cross1)) * \
               np.sqrt(np.sum(vec2**2))
        arg2 = np.sum(np.multiply(cross1, cross2))
        answer = np.arctan2(arg1, arg2)
        return answer
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a
        o = self.b
        p = self.c
        n = self.d
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        if (1 - np.dot(u, w)**2) < 1e-6:
            term1 = np.cross(u, w) * 0
            term3 = np.cross(u, w) * 0
        else:
            term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
            term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        if (1 - np.dot(v, w)**2) < 1e-6:
            term2 = np.cross(v, w) * 0
            term4 = np.cross(v, w) * 0
        else:
            term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
            term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        # term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        # term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        # term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        derivatives[m, :] = term1
        derivatives[n, :] = -term2
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives

class InternalCoordinates(object):
    def wilsonB(self, xyz):
        """
        Given Cartesian coordinates xyz, return the Wilson B-matrix
        given by dq_i/dx_j where x is flattened (i.e. x1, y1, z1, x2, y2, z2)
        """
        WilsonB = []
        Der = self.derivatives(xyz)
        for i in range(Der.shape[0]):
            WilsonB.append(Der[i].flatten())
        return np.array(WilsonB)

    def GMatrix(self, xyz, u=None):
        """
        Given Cartesian coordinates xyz, return the G-matrix
        given by G = BuBt where u is an arbitrary matrix (default to identity)
        """
        Bmat = np.matrix(self.wilsonB(xyz))
        if u is None:
            BuBt = Bmat*Bmat.T
        else:
            BuBt = Bmat * u * Bmat.T
        return BuBt

    def GInverse_SVD(self, xyz, u=None):
        xyz = xyz.reshape(-1,3)
        # Perform singular value decomposition
        click()
        G = self.GMatrix(xyz, u)
        time_G = click()
        U, S, VT = np.linalg.svd(G)
        time_svd = click()
        # print "Build G: %.3f SVD: %.3f" % (time_G, time_svd),
        V = np.matrix(VT).T
        UT = np.matrix(U).T
        Sinv = np.zeros_like(S)
        LargeVals = 0
        for ival, value in enumerate(S):
            # print "% .5e" % value
            if np.abs(value) > 1e-6:
                LargeVals += 1
                Sinv[ival] = 1/value
        # print "%i atoms; %i/%i singular values are > 1e-6" % (xyz.shape[0], LargeVals, len(S))
        Sinv = np.matrix(np.diag(Sinv))
        Inv = np.matrix(V)*Sinv*np.matrix(UT)
        return np.matrix(V)*Sinv*np.matrix(UT)

    def GInverse_EIG(self, xyz, u=None):
        xyz = xyz.reshape(-1,3)
        click()
        G = self.GMatrix(xyz, u)
        time_G = click()
        Gi = np.linalg.inv(G)
        time_inv = click()
        # print "G-time: %.3f Inv-time: %.3f" % (time_G, time_inv)
        return Gi

    def checkFiniteDifference(self, xyz):
        xyz = xyz.reshape(-1,3)
        Analytical = self.derivatives(xyz)
        FiniteDifference = np.zeros_like(Analytical)
        h = 0.001
        for i in range(xyz.shape[0]):
            for j in range(3):
                x1 = xyz.copy()
                x2 = xyz.copy()
                x1[i,j] += h
                x2[i,j] -= h
                PMDiff = self.calcDiff(x1,x2)
                FiniteDifference[:,i,j] = PMDiff/(2*h)
        for i in range(Analytical.shape[0]):
            print "IC %i/%i :" % (i, Analytical.shape[0])
            for j in range(Analytical.shape[1]):
                print "Atom %i" % (j+1)
                for k in range(Analytical.shape[2]):
                    print "xyz"[k],
                    error = Analytical[i,j,k] - FiniteDifference[i,j,k]
                    if np.abs(error) > 1e-5:
                        color = "\x1b[91m"
                    else:
                        color = "\x1b[92m"
                    # if np.abs(error) > 1e-5:
                    print "% .5e % .5e %s% .5e\x1b[0m" % (Analytical[i,j,k], FiniteDifference[i,j,k], color, Analytical[i,j,k] - FiniteDifference[i,j,k])
        print "Finite-difference Finished"

    def calcGrad(self, xyz, gradx):
        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
        return np.array(Gq).flatten()

    def newCartesian(self, xyz, dQ, u=None, verbose=False):
        xyz1 = xyz.copy()
        dQ1 = dQ.copy()
        # Iterate until convergence:
        microiter = 0
        rmsds = []
        self.bork = False
        # Damping factor
        TB = 1.0
        while True:
            microiter += 1
            if microiter == 10:
                self.bork = True
                if verbose: print "Approximate coordinates obtained after %i microiterations (rmsd = %.3f)" % (microiter, rmsds[0])
                return xyzsave.flatten()
            Bmat = np.matrix(self.wilsonB(xyz1))
            Ginv = self.GInverse(xyz1, u)
            # Get new Cartesian coordinates
            if u is not None:
                dxyz = TB*u*Bmat.T*Ginv*(np.matrix(dQ1).T)
            else:
                dxyz = TB*Bmat.T*Ginv*(np.matrix(dQ1).T)
            xyz2 = xyz1 + np.array(dxyz).flatten()
            if microiter == 1:
                xyzsave = xyz2.copy()
            rmsd = np.sqrt(np.mean((np.array(xyz2-xyz1).flatten())**2))
            if len(rmsds) > 0:
                if rmsd > rmsdt:
                    if verbose: print "Iter: %i RMSD: %.3e Thre: % .3e Damp: %.3f (Bad)" % (microiter, rmsd, rmsdt, TB)
                    TB /= 2
                else:
                    if verbose: print "Iter: %i RMSD: %.3e Thre: % .3e Damp: %.3f (Good)" % (microiter, rmsd, rmsdt, TB)
                    TB = min(TB*1.25, 1.0)
                    rmsdt = rmsd
            else:
                if verbose: print "Iter: %i RMSD: %.3e" % (microiter, rmsd)
                rmsdt = rmsd
            rmsds.append(rmsd)
            # Are we converged?
            if rmsd < 1e-6:
                if verbose: print "Cartesian coordinates obtained after %i microiterations" % microiter
                break
            # Calculate the actual change in internal coordinates
            dQ_actual = self.calcDiff(xyz2, xyz1)
            # Figure out the further change needed
            dQ1 -= dQ_actual
            xyz1 = xyz2
        return xyz2.flatten()
    
class RedundantInternalCoordinates(InternalCoordinates):
    def __repr__(self):
        lines = ["Internal coordinate system (atoms numbered from 1):"]
        typedict = OrderedDict()
        for Internal in self.Internals:
            lines.append(Internal.__repr__())
            if str(type(Internal)) not in typedict:
                typedict[str(type(Internal))] = 1
            else:
                typedict[str(type(Internal))] += 1
        if len(lines) > 100:
            # Print only summary if too many
            lines = []
        for k, v in typedict.items():
            lines.append("%s : %i" % (k, v))
        return '\n'.join(lines)

    def __eq__(self, other):
        answer = True
        for i in self.Internals:
            if i not in other.Internals:
                answer = False
        for i in other.Internals:
            if i not in self.Internals:
                answer = False
        return answer

    def update(self, other):
        Changed = False
        for i in self.Internals:
            if i not in other.Internals:
                if hasattr(i, 'inactive'):
                    i.inactive += 1
                else:
                    i.inactive = 0
                if i.inactive == 1:
                    print "Deleting:", i
                    self.Internals.remove(i)
                    Changed = True
            else:
                i.inactive = 0
        for i in other.Internals:
            if i not in self.Internals:
                print "Adding:  ", i
                self.Internals.append(i)
                Changed = True
        return Changed

    def __ne__(self, other):
        return not self.__eq__(other)

    def repr_diff(self, other):
        alines = ["-- Added: --"]
        for i in other.Internals:
            if i not in self.Internals:
                alines.append(i.__repr__())
        dlines = ["-- Deleted: --"]
        for i in self.Internals:
            if i not in other.Internals:
                dlines.append(i.__repr__())
        output = []
        if len(alines) > 1:
            output += alines
        if len(dlines) > 1:
            output += dlines
        return '\n'.join(output)

    def calculate(self, xyz):
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.value(xyz))
        return np.array(answer)

    def calculateDegrees(self, xyz):
        answer = []
        for Internal in self.Internals:
            value = Internal.value(xyz)
            if type(Internal) in [Angle, Dihedral, OutOfPlane]:
                value *= 180/np.pi
            answer.append(value)
        return np.array(answer)

    def derivatives(self, xyz):
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        return np.array(answer)

    def calcDiff(self, coord1, coord2):
        """ Calculate difference in internal coordinates, accounting for changes in 2*pi of angles. """
        Q1 = self.calculate(coord1)
        Q2 = self.calculate(coord2)
        PMDiff = (Q1-Q2)
        for k in range(len(PMDiff)):
            if type(self.Internals[k]) in [Angle, Dihedral, OutOfPlane]:
                Plus2Pi = PMDiff[k] + 2*np.pi
                Minus2Pi = PMDiff[k] - 2*np.pi
                if np.abs(PMDiff[k]) > np.abs(Plus2Pi):
                    PMDiff[k] = Plus2Pi
                if np.abs(PMDiff[k]) > np.abs(Minus2Pi):
                    PMDiff[k] = Minus2Pi
        return PMDiff

    def GInverse(self, xyz, u=None):
        return self.GInverse_SVD(xyz, u)

    def addMultiCartesianX(self, i, w=1.0):
        Cart = MultiCartesianX(i, w)
        if Cart not in self.Internals:
            self.Internals.append(Cart)

    def addMultiCartesianY(self, i, w=1.0):
        Cart = MultiCartesianY(i, w)
        if Cart not in self.Internals:
            self.Internals.append(Cart)

    def addMultiCartesianZ(self, i, w=1.0):
        Cart = MultiCartesianZ(i, w)
        if Cart not in self.Internals:
            self.Internals.append(Cart)

    def addCartesianX(self, i, w=1.0):
        Cart = CartesianX(i, w)
        if Cart not in self.Internals:
            self.Internals.append(Cart)

    def addCartesianY(self, i, w=1.0):
        Cart = CartesianY(i, w)
        if Cart not in self.Internals:
            self.Internals.append(Cart)

    def addCartesianZ(self, i, w=1.0):
        Cart = CartesianZ(i, w)
        if Cart not in self.Internals:
            self.Internals.append(Cart)
    
    def addDistance(self, i, j):
        # Ideally these three functions could somehow be generalized into one,
        # because I think we could develop better internal coordinates with
        # solid geometry
        Dist = Distance(i,j)
        if Dist not in self.Internals:
            self.Internals.append(Dist)
            
    def addAngle(self, i, j, k):
        Ang = Angle(i,j,k)
        if Ang not in self.Internals:
            self.Internals.append(Ang)

    def addDihedral(self, i, j, k, l):
        Dih = Dihedral(i,j,k,l)
        if Dih not in self.Internals:
            self.Internals.append(Dih)

    def addOutOfPlane(self, i, j, k, l):
        Oop = OutOfPlane(i,j,k,l)
        if Oop not in self.Internals:
            self.Internals.append(Oop)

    def delMultiCartesianX(self, i):
        Cart = MultiCartesianX(i)
        for ii in range(len(self.Internals))[::-1]:
            if Cart == self.Internals[ii]:
                del self.Internals[ii]

    def delMultiCartesianY(self, i):
        Cart = MultiCartesianY(i)
        for ii in range(len(self.Internals))[::-1]:
            if Cart == self.Internals[ii]:
                del self.Internals[ii]

    def delMultiCartesianZ(self, i):
        Cart = MultiCartesianZ(i)
        for ii in range(len(self.Internals))[::-1]:
            if Cart == self.Internals[ii]:
                del self.Internals[ii]

    def delCartesianX(self, i):
        Cart = CartesianX(i)
        for ii in range(len(self.Internals))[::-1]:
            if Cart == self.Internals[ii]:
                del self.Internals[ii]

    def delCartesianY(self, i):
        Cart = CartesianY(i)
        for ii in range(len(self.Internals))[::-1]:
            if Cart == self.Internals[ii]:
                del self.Internals[ii]

    def delCartesianZ(self, i):
        Cart = CartesianZ(i)
        for ii in range(len(self.Internals))[::-1]:
            if Cart == self.Internals[ii]:
                del self.Internals[ii]
            
    def delDistance(self, i, j):
        Dist = Distance(i, j)
        for ii in range(len(self.Internals))[::-1]:
            if Dist == self.Internals[ii]:
                del self.Internals[ii]

    def delAngle(self, i, j, k):
        Ang = Angle(i, j, k)
        for ii in range(len(self.Internals))[::-1]:
            if Ang == self.Internals[ii]:
                del self.Internals[ii]

    def delDihedral(self, i, j, k, l):
        Dih = Dihedral(i, j, k, l)
        for ii in range(len(self.Internals))[::-1]:
            if Dih == self.Internals[ii]:
                del self.Internals[ii]

    def delOutOfPlane(self, i, j, k, l):
        Oop = OutOfPlane(i, j, k, l)
        for ii in range(len(self.Internals))[::-1]:
            if Oop == self.Internals[ii]:
                del self.Internals[ii]
                
    def __init__(self, molecule, connect=False):
        self.connect = connect
        self.Internals = []
        self.elem = molecule.elem
        if len(molecule) != 1:
            raise RuntimeError('Only one frame allowed in molecule object')
        # Determine the atomic connectivity
        molecule.build_topology(Fac=1.3)
        frags = [m.nodes() for m in molecule.molecules]
        # Coordinates in Angstrom
        coords = molecule.xyzs[0].flatten()

        # Make a distance matrix mapping atom pairs to interatomic distances
        AtomIterator, dxij = molecule.distance_matrix()
        D = {}
        for i, j in zip(AtomIterator, dxij[0]):
            assert i[0] < i[1]
            D[tuple(i)] = j
        dgraph = nx.Graph()
        for i in range(molecule.na):
            dgraph.add_node(i)
        for k, v in D.items():
            dgraph.add_edge(k[0], k[1], weight=v)
        mst = sorted(list(nx.minimum_spanning_edges(dgraph, data=False)))
        # Build a list of noncovalent distances
        noncov = []
        # Connect all non-bonded fragments together
        for edge in mst:
            if edge not in list(molecule.topology.edges()):
                # print "Adding %s from minimum spanning tree" % str(edge)
                if connect:
                    molecule.topology.add_edge(edge[0], edge[1])
                    noncov.append(edge)
        if not connect:
            for i in frags:
                self.addMultiCartesianX(i, w=1e-3/len(i))
                self.addMultiCartesianY(i, w=1e-3/len(i))
                self.addMultiCartesianZ(i, w=1e-3/len(i))
            for i in range(molecule.na):
                self.addCartesianX(i, w=1)
                self.addCartesianY(i, w=1)
                self.addCartesianZ(i, w=1)

        # # Build a list of noncovalent distances
        # noncov = []
        # # Connect all non-bonded fragments together
        # while True:
        #     # List of disconnected fragments
        #     subg = list(nx.connected_component_subgraphs(molecule.topology))
        #     # Break out of loop if all fragments are connected
        #     if len(subg) == 1: break
        #     # Find the smallest interatomic distance between any pair of fragments
        #     minD = 1e10
        #     for i in range(len(subg)):
        #         for j in range(i):
        #             for a in subg[i].nodes():
        #                 for b in subg[j].nodes():
        #                     if D[(min(a,b), max(a,b))] < minD:
        #                         minD = D[(min(a,b), max(a,b))]
        #     # Next, create one connection between pairs of fragments that have a
        #     # close-contact distance of at most 1.2 times the minimum found above
        #     for i in range(len(subg)):
        #         for j in range(i):
        #             tminD = 1e10
        #             conn = False
        #             conn_a = None
        #             conn_b = None
        #             for a in subg[i].nodes():
        #                 for b in subg[j].nodes():
        #                     if D[(min(a,b), max(a,b))] < tminD:
        #                         tminD = D[(min(a,b), max(a,b))]
        #                         conn_a = min(a,b)
        #                         conn_b = max(a,b)
        #                     if D[(min(a,b), max(a,b))] <= 1.3*minD:
        #                         conn = True
        #             if conn:
        #                 molecule.topology.add_edge(conn_a, conn_b)
        #                 noncov.append((conn_a, conn_b))

        # Add an internal coordinate for all interatomic distances
        for (a, b) in molecule.topology.edges():
            self.addDistance(a, b)

        # Add an internal coordinate for all angles
        LinThre = 0.99619469809174555
        # LinThre = 0.999
        AngDict = defaultdict(list)
        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    if a < c:
                        Ang = Angle(a, b, c)
                        nnc = (min(a, b), max(a, b)) in noncov
                        nnc += (min(b, c), max(b, c)) in noncov
                        # if nnc >= 2: continue
                        if np.abs(np.cos(Ang.value(coords))) < LinThre:
                            self.addAngle(a, b, c)
                            AngDict[b].append(Ang)
                        elif self.connect:
                            # print Ang, "is linear: replacing with Cartesians"
                            # Almost linear bends (greater than 175 or less than 5) are dropped. 
                            # The dropped angle is replaced by the two Cartesians of the central 
                            # atom that are most perpendicular to the line between the other two 
                            # atoms forming the bend (as measured by the corresponding scalar 
                            # products with the Cartesian axes).
                            # if len(molecule.topology.neighbors(b)) == 2:
                            # Unit vector connecting atoms a and c
                            nac = molecule.xyzs[0][c] - molecule.xyzs[0][a]
                            nac /= np.linalg.norm(nac)
                            # Dot products of this vector with the Cartesian axes
                            dots = [np.dot(ei, nac) for ei in np.eye(3)]
                            # Functions for adding Cartesian coordinate
                            carts = [self.addCartesianX, self.addCartesianY, self.addCartesianZ]
                            # Add two of the most perpendicular Cartesian coordinates
                            for i in np.argsort(dots)[:2]:
                                carts[i](b)

        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    for d in molecule.topology.neighbors(b):
                        if a < c < d:
                            nnc = (min(a, b), max(a, b)) in noncov
                            nnc += (min(b, c), max(b, c)) in noncov
                            nnc += (min(b, d), max(b, d)) in noncov
                            # if nnc >= 1: continue
                            for i, j, k in list(itertools.permutations([a, c, d], 3)):
                                Ang1 = Angle(b,i,j)
                                Ang2 = Angle(i,j,k)
                                if np.abs(np.cos(Ang1.value(coords))) > LinThre: continue
                                if np.abs(np.cos(Ang2.value(coords))) > LinThre: continue
                                if np.abs(np.dot(Ang1.normal_vector(coords), Ang2.normal_vector(coords))) > 0.95:
                                    self.delAngle(i, b, j)
                                    self.addOutOfPlane(b, i, j, k)
                                    break
                                
        # Lines-of-atoms code, commented out for now
        # atom_lines = [list(i) for i in molecule.topology.edges()]
        # while True:
        #     atom_lines0 = deepcopy(atom_lines)
        #     for aline in atom_lines:
        #         # Imagine a line of atoms going like ab-ac-ax-ay.
        #         # Our job is to extend the line until there are no more
        #         ab = aline[0]
        #         ac = aline[1]
        #         ax = aline[-2]
        #         ay = aline[-1]
        #         for aa in molecule.topology.neighbors(ab):
        #             if aa not in aline:
        #                 ang = Angle(aa, ab, ac)
        #                 if np.abs(np.cos(ang.value(coords))) > LinThre:
        #                     # print "Prepending", aa, "to", aline
        #                     aline.insert(0, aa)
        #         for az in molecule.topology.neighbors(ay):
        #             if az not in aline:
        #                 ang = Angle(ax, ay, az)
        #                 if np.abs(np.cos(ang.value(coords))) > LinThre:
        #                     # print "Appending", az, "to", aline
        #                     aline.append(az)
        #     if atom_lines == atom_lines0: break
        # atom_lines_uniq = []
        # for i in atom_lines:
        #     if tuple(i) not in set(atom_lines_uniq):
        #         atom_lines_uniq.append(tuple(i))
        # print "Lines of atoms:", atom_lines_uniq

        for (b, c) in molecule.topology.edges():
        # for aline in atom_lines_uniq:
        #     b = aline[0]
        #     c = aline[-1]
            for a in molecule.topology.neighbors(b):
                for d in molecule.topology.neighbors(c):
                    # if a not in aline and d not in aline and a != d:
                    if a != c and b != d and a != d:
                        nnc = (min(a, b), max(a, b)) in noncov
                        nnc += (min(b, c), max(b, c)) in noncov
                        nnc += (min(c, d), max(c, d)) in noncov
                        # if nnc >= 2: continue
                        Ang1 = Angle(a,b,c)
                        Ang2 = Angle(b,c,d)
                        if np.abs(np.cos(Ang1.value(coords))) > LinThre: continue
                        if np.abs(np.cos(Ang2.value(coords))) > LinThre: continue
                        self.addDihedral(a, b, c, d)

    def guess_hessian(self, coords):
        xyzs = coords.reshape(-1,3)*0.520
        Hdiag = []
        def covalent(a, b):
            r = np.linalg.norm(xyzs[a]-xyzs[b])
            rcov = Radii[Elements.index(self.elem[a])-1] + Radii[Elements.index(self.elem[b])-1]
            return r/rcov < 1.2
        
        for ic in self.Internals:
            if type(ic) is Distance:
                r = np.linalg.norm(xyzs[ic.a]-xyzs[ic.b]) / 0.529
                elem1 = min(Elements.index(self.elem[ic.a]), Elements.index(self.elem[ic.b]))
                elem2 = max(Elements.index(self.elem[ic.a]), Elements.index(self.elem[ic.b]))
                A = 1.734
                if elem1 < 3:
                    if elem2 < 3:
                        B = -0.244
                    elif elem2 < 11:
                        B = 0.352
                    else:
                        B = 0.660
                elif elem1 < 11:
                    if elem2 < 11:
                        B = 1.085
                    else:
                        B = 1.522
                else:
                    B = 2.068
                if covalent(ic.a, ic.b):
                    Hdiag.append(A/(r-B)**3)
                else:
                    Hdiag.append(0.1)
            elif type(ic) is Angle:
                if min(Elements.index(self.elem[ic.a]),
                       Elements.index(self.elem[ic.b]),
                       Elements.index(self.elem[ic.c])) < 3:
                    A = 0.160
                else:
                    A = 0.250
                if covalent(ic.a, ic.b) and covalent(ic.b, ic.c):
                    Hdiag.append(A)
                else:
                    Hdiag.append(0.1)
            elif type(ic) is Dihedral:
                r = np.linalg.norm(xyzs[ic.b]-xyzs[ic.c])
                rcov = Radii[Elements.index(self.elem[ic.b])-1] + Radii[Elements.index(self.elem[ic.c])-1]
                Hdiag.append(0.1)
                # print r, rcov
                # Hdiag.append(0.0023 - 0.07*(r-rcov))
            elif type(ic) is OutOfPlane:
                r1 = xyzs[ic.b]-xyzs[ic.a]
                r2 = xyzs[ic.c]-xyzs[ic.a]
                r3 = xyzs[ic.d]-xyzs[ic.a]
                d = 1 - np.abs(np.dot(r1,np.cross(r2,r3))/np.linalg.norm(r1)/np.linalg.norm(r2)/np.linalg.norm(r3))
                Hdiag.append(0.1)
                # These formulas appear to be useless
                # if covalent(ic.a, ic.b) and covalent(ic.a, ic.c) and covalent(ic.a, ic.d):
                #     Hdiag.append(0.045)
                # else:
                #     Hdiag.append(0.023)
            elif type(ic) in [CartesianX, CartesianY, CartesianZ]:
                Hdiag.append(0.01)
            elif type(ic) in [MultiCartesianX, MultiCartesianY, MultiCartesianZ]:
                Hdiag.append(0.01)
            else:
                raise RuntimeError('Spoo!')
        return np.matrix(np.diag(Hdiag))

class DelocalizedInternalCoordinates(InternalCoordinates):
    def __init__(self, molecule, build=True, connect=False):
        self.Prims = RedundantInternalCoordinates(molecule, connect)
        self.connect = connect
        xyz = molecule.xyzs[0].flatten() / 0.529
        self.na = molecule.na
        if build:
            self.build_dlc(xyz)

    def __repr__(self):
        return self.Prims.__repr__()
            
    def update(self, other):
        return self.Prims.update(other.Prims)

    def build_dlc(self, xyz):
        # Perform singular value decomposition
        click()
        G = self.Prims.GMatrix(xyz)
        time_G = click()
        L, Q = np.linalg.eigh(G)
        time_eig = click()
        # print "Build G: %.3f Eig: %.3f" % (time_G, time_eig)
        LargeVals = 0
        LargeIdx = []
        for ival, value in enumerate(L):
            # print value
            if np.abs(value) > 1e-6:
                LargeVals += 1
                LargeIdx.append(ival)
        Expect = 3*self.na - 6
        if self.na == 2:
            Expect = 1
        if self.na == 1:
            Expect = 0
        print "%i atoms (expect %i coordinates); %i/%i singular values are > 1e-6" % (self.na, Expect, LargeVals, len(L))
        # if LargeVals <= Expect:
        self.Vecs = Q[:, LargeIdx]
        self.Internals = ["DLC %i" % (i+1) for i in range(len(LargeIdx))]
        # else:
        #     Idxs = np.argsort(L)[-Expect:]
        #     self.Vecs = Q[:, Idxs]
        #     self.Internals = ["DLC %i" % (i+1) for i in range(Expect)]

    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)

    def calcDiff(self, coord1, coord2):
        """ Calculate difference in internal coordinates, accounting for changes in 2*pi of angles. """
        PMDiff = self.Prims.calcDiff(coord1, coord2)
        Answer = np.matrix(PMDiff)*self.Vecs
        return np.array(Answer).flatten()

    def calculate(self, coords):
        PrimVals = self.Prims.calculate(coords)
        Answer = np.matrix(PrimVals)*self.Vecs
        return np.array(Answer).flatten()

    def derivatives(self, coords):
        PrimDers = self.Prims.derivatives(coords)
        # The following code does the same as "tensordot"
        # print PrimDers.shape
        # print self.Vecs.shape
        # Answer = np.zeros((self.Vecs.shape[1], PrimDers.shape[1], PrimDers.shape[2]), dtype=float)
        # for i in range(self.Vecs.shape[1]):
        #     for j in range(self.Vecs.shape[0]):
        #         Answer[i, :, :] += self.Vecs[j, i] * PrimDers[j, :, :]
        # print Answer.shape
        Answer1 = np.tensordot(self.Vecs, PrimDers, axes=(0, 0))
        return np.array(Answer1)

    def GInverse(self, xyz, u=None):
        return self.GInverse_EIG(xyz, u)

    def repr_diff(self, other):
        return self.Prims.repr_diff(other.Prims)

    def guess_hessian(self, coords):
        Hprim = np.matrix(self.Prims.guess_hessian(coords))
        return np.array(self.Vecs.T*Hprim*self.Vecs)
