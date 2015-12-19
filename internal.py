#!/usr/bin/env python

from __future__ import division
import numpy as np
import networkx as nx
from forcebalance.nifty import click
from collections import OrderedDict

class CartesianX(object):
    def __init__(self, a):
        self.a = a

    def __repr__(self):
        return "Cartesian-X %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return self.a == other.a

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][0]
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives[a][0] = 1.0
        return derivatives

class CartesianY(object):
    def __init__(self, a):
        self.a = a

    def __repr__(self):
        return "Cartesian-Y %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return self.a == other.a

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][1]
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives[a][1] = 1.0
        return derivatives

class CartesianZ(object):
    def __init__(self, a):
        self.a = a

    def __repr__(self):
        return "Cartesian-Z %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return self.a == other.a

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][2]
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives[a][2] = 1.0
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
        # if (1 - np.dot(u, w)**2) < 1e-6:
        #     term1 = np.cross(u, w) * 0
        #     term3 = np.cross(u, w) * 0
        # else:
        #     term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        #     term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # if (1 - np.dot(v, w)**2) < 1e-6:
        #     term2 = np.cross(v, w) * 0
        #     term4 = np.cross(v, w) * 0
        # else:
        #     term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        #     term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
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
        # if (1 - np.dot(u, w)**2) < 1e-6:
        #     term1 = np.cross(u, w) * 0
        #     term3 = np.cross(u, w) * 0
        # else:
        #     term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        #     term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # if (1 - np.dot(v, w)**2) < 1e-6:
        #     term2 = np.cross(v, w) * 0
        #     term4 = np.cross(v, w) * 0
        # else:
        #     term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        #     term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        derivatives[m, :] = term1
        derivatives[n, :] = -term2
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives
    
class InternalCoordinates(object):
    def __init__(self):
        self.Internals = []

    def __repr__(self):
        lines = ["Internal coordinate system (atoms numbered from 1):"]
        typedict = OrderedDict()
        for Internal in self.Internals:
            lines.append(Internal.__repr__())
            if str(type(Internal)) not in typedict:
                typedict[str(type(Internal))] = 0
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
                # print i, "not in the other"
                answer = False
        for i in other.Internals:
            if i not in self.Internals:
                # print i, "not in self"
                answer = False
        return answer

    def __ne__(self, other):
        return not self.__eq__(other)

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
        print "Build G: %.3f SVD: %.3f" % (time_G, time_svd),
        V = np.matrix(VT).T
        UT = np.matrix(U).T
        Sinv = np.zeros_like(S)
        # Tikhonov regularization
        # Sigma = 1e-6
        LargeVals = 0
        for ival, value in enumerate(S):
            # print "% .5e" % value
            if np.abs(value) > 1e-6:
                LargeVals += 1
                Sinv[ival] = 1/value
            # Sinv[ival] = value / (value**2 + Sigma**2)
        # print "%i atoms; %i/%i singular values are > 1e-6" % (xyz.shape[0], LargeVals, len(S))
        Sinv = np.matrix(np.diag(Sinv))
        Inv = np.matrix(V)*Sinv*np.matrix(UT)
        # print np.max(np.abs(G*Inv-np.eye(G.shape[0])))
        # raw_input()
        return np.matrix(V)*Sinv*np.matrix(UT)

    def GInverse(self, xyz, u=None):
        SVD = True
        if SVD:
            return self.GInverse_SVD(xyz, u)
        else:
            # Warning - this code won't work because the matrix is ill-conditioned
            xyz = xyz.reshape(-1,3)
            # Perform singular value decomposition
            click()
            G = self.GMatrix(xyz, u)
            time_G = click()
            Gi = np.linalg.pinv(G,rcond=1e-10)
            time_inv = click()
            print "G-time: %.3f Pinv-time: %.3f" % (time_G, time_inv)
            return Gi
            print L
            raw_input()
            Q = np.matrix(Q)
            Linv = np.zeros_like(L)
            # Tikhonov regularization
            # Sigma = 1e-6
            LargeVals = 0
            for ival, value in enumerate(L):
                # print "% .5e" % value
                if np.abs(value) > 1e-6:
                    LargeVals += 1
                    Linv[ival] = 1/value
                # Sinv[ival] = value / (value**2 + Sigma**2)
            # print "%i atoms; %i/%i singular values are > 1e-6" % (xyz.shape[0], LargeVals, len(S))
            Linv = np.matrix(np.diag(L))
            Inv = Q*Linv*Q.T
            # print np.max(np.abs(Q*Q.T-np.eye(Q.shape[0])))
            # print np.max(np.abs(G*Inv-np.eye(G.shape[0])))
            # raw_input()
            return Inv

    def calcGrad(self, xyz, gradx):
        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
        return np.array(Gq).flatten()

    def newCartesian(self, xyz, dQ, u=None):
        xyz1 = xyz.copy()
        dQ1 = dQ.copy()
        # Iterate until convergence:
        microiter = 0
        rmsds = []
        self.bork = False
        while True:
            microiter += 1
            if microiter == 10:
                self.bork = True
                print "Approximate coordinates obtained after %i microiterations (rmsd = %.3f)" % (microiter, rmsds[0])
                return xyzsave.flatten()
            # if microiter > 3 and rmsds[-1] > rmsds[-2]:
            #     print "Approximate coordinates obtained after %i microiterations (rmsd = %.3f)" % (microiter, rmsds[0])
            #     self.bork = True
            #     return xyzsave.flatten()
            # Calculate Bt, Ginv
            Bmat = np.matrix(self.wilsonB(xyz1))
            Ginv = self.GInverse(xyz1, u)
            # Get new Cartesian coordinates
            if u is not None:
                dxyz = u*Bmat.T*Ginv*(np.matrix(dQ1).T)
            else:
                dxyz = Bmat.T*Ginv*(np.matrix(dQ1).T)
            xyz2 = xyz1 + np.array(dxyz).flatten()
            if microiter == 1:
                xyzsave = xyz2.copy()
            # Are we coverged?
            rmsd = np.sqrt(np.mean((np.array(xyz2-xyz1).flatten())**2))
            rmsds.append(rmsd)
            print "Iter: %i RMSD: %.3e" % (microiter, rmsd)
            # print microiter, rmsd
            if rmsd < 1e-6:
                print "Cartesian coordinates obtained after %i microiterations" % microiter
                break
            # Calculate the actual change in internal coordinates
            Q1 = self.calculate(xyz1)
            Q2 = self.calculate(xyz2)
            dQ_actual = self.subtractInternal(Q2,Q1)
            # Figure out the further change needed
            dQ1 -= dQ_actual
            xyz1 = xyz2
        return xyz2.flatten()

    def subtractInternal(self, Q1, Q2):
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

    def checkFiniteDifference(self, xyz):
        xyz = xyz.reshape(-1,3)
        Analytical = self.derivatives(xyz)
        FiniteDifference = np.zeros_like(Analytical)
        ZeroPoint = self.calculate(xyz)
        h = 0.001
        for i in range(xyz.shape[0]):
            for j in range(3):
                xyz[i,j] += h
                PlusPoint = self.calculate(xyz)
                xyz[i,j] -= 2*h
                MinusPoint = self.calculate(xyz)
                xyz[i,j] += h
                PMDiff = self.subtractInternal(PlusPoint, MinusPoint)
                FiniteDifference[:,i,j] = PMDiff/(2*h)
        for i in range(Analytical.shape[0]):
            print "IC %i/%i :" % (i, Analytical.shape[0]), self.Internals[i]
            for j in range(Analytical.shape[1]):
                print "Atom %i" % (j+1)
                for k in range(Analytical.shape[2]):
                    print "xyz"[k],
                    error = Analytical[i,j,k] - FiniteDifference[i,j,k]
                    if np.abs(error) > 1e-5:
                        color = "\x1b[91m"
                    else:
                        color = "\x1b[92m"
                    if np.abs(error) > 1e-5:
                        print "% .5e % .5e %s% .5e\x1b[0m" % (Analytical[i,j,k], FiniteDifference[i,j,k], color, Analytical[i,j,k] - FiniteDifference[i,j,k])
        print "Finite-difference Finished"
    
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
                
    @classmethod
    def buildFromMolecule(cls, molecule):
        obj = cls()
        if len(molecule) != 1:
            raise RuntimeError('Only one frame allowed in molecule object')
        # Determine the atomic connectivity
        molecule.build_topology(Fac=1.25)
        # Make a distance matrix mapping atom pairs to interatomic distances
        AtomIterator, dxij = molecule.distance_matrix()
        D = {}
        for i, j in zip(AtomIterator, dxij[0]):
            assert i[0] < i[1]
            D[tuple(i)] = j
        # Build a list of noncovalent distances
        noncov = []
        # Connect all non-bonded fragments together
        while True:
            subg = list(nx.connected_component_subgraphs(molecule.topology))
            if len(subg) == 1: break
            minD = 1e10
            for i in range(len(subg)):
                for j in range(i):
                    for a in subg[i].nodes():
                        for b in subg[j].nodes():
                            if D[(min(a,b), max(a,b))] < minD:
                                minD = D[(min(a,b), max(a,b))]
            for i in range(len(subg)):
                for j in range(i):
                    tminD = 1e10
                    connect = False
                    conn_a = None
                    conn_b = None
                    for a in subg[i].nodes():
                        for b in subg[j].nodes():
                            if D[(min(a,b), max(a,b))] < tminD:
                                tminD = D[(min(a,b), max(a,b))]
                                conn_a = min(a,b)
                                conn_b = max(a,b)
                            if D[(min(a,b), max(a,b))] <= 1.25*minD:
                                connect = True
                    if connect:
                        molecule.topology.add_edge(conn_a, conn_b)
                        noncov.append((conn_a, conn_b))
                                # print "Connecting atoms %i-%i with distance %.3f" % (min(a,b),max(a,b),D[(min(a,b),max(a,b))])

        for (a, b) in molecule.topology.edges():
            obj.addDistance(a, b)
        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    if a < c:
                        # nnoncov = (min(a,b),max(a,b)) in noncov
                        # nnoncov += (min(b,c),max(b,c)) in noncov
                        # if nnoncov > 1: continue
                        obj.addAngle(a, b, c)
        OopAngs = []
        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    for d in molecule.topology.neighbors(b):
                        if a < c < d:
                            # Do not make out-of-plane for noncovalent interactions
                            if (min(a,b),max(a,b)) in noncov: continue 
                            if (min(a,c),max(a,c)) in noncov: continue
                            if (min(a,d),max(a,d)) in noncov: continue
                            Ang1 = Angle(b,a,c)
                            Ang2 = Angle(a,c,d)
                            LinearAngle = False
                            OopAngs.append((a, b, c))
                            OopAngs.append((a, b, d))
                            OopAngs.append((c, b, d))
                            # if np.abs(np.sin(Ang1.value(coords))) < 0.05:
                            #     LinearAngle = True
                            # if np.abs(np.sin(Ang2.value(coords))) < 0.05:
                            #     LinearAngle = True
                            if LinearAngle:
                                print "Out-of-plane %i-%i-%i-%i contains a linear angle" % (b, a, c, d)
                            else:
                                obj.addOutOfPlane(b, a, c, d)
        for (b, c) in molecule.topology.edges():
            for a in molecule.topology.neighbors(b):
                for d in molecule.topology.neighbors(c):
                    if a != c and b != d and a != d:
                        nnoncov = (min(a,b),max(a,b)) in noncov
                        nnoncov += (min(b,c),max(b,c)) in noncov
                        nnoncov += (min(c,d),max(c,d)) in noncov
                        if nnoncov > 1: continue
                        Ang1 = Angle(a,b,c)
                        Ang2 = Angle(b,c,d)
                        LinearAngle = False
                        if a < c:
                            Ang1 = (a, b, c)
                        else:
                            Ang1 = (c, b, a)
                        if b < d:
                            Ang2 = (b, c, d)
                        else:
                            Ang2 = (d, c, b)
                        # # print OopAngs, Ang1, Ang2
                        # # raw_input()
                        # print molecule.elem[a], molecule.elem[b], molecule.elem[c], molecule.elem[d], Ang1, Ang2, Ang1 in OopAngs, Ang2 in OopAngs
                        if Ang1 in OopAngs and Ang2 in OopAngs:
                            continue
                        # if np.abs(np.sin(Ang1.value(coords))) < 0.05:
                        #     LinearAngle = True
                        # if np.abs(np.sin(Ang2.value(coords))) < 0.05:
                        #     LinearAngle = True
                        if LinearAngle:
                            print "Dihedral %i-%i-%i-%i contains a linear angle" % (a, b, c, d)
                        else:
                            obj.addDihedral(a, b, c, d)
        return obj
    
