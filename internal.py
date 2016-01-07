#!/usr/bin/env python

from __future__ import division
import numpy as np
import networkx as nx
import itertools
from copy import deepcopy
from forcebalance.nifty import click, invert_svd, commadash, row, col, flat
from forcebalance.molecule import Molecule, Elements, Radii
from collections import OrderedDict, defaultdict
from scipy import optimize
from rotate import get_expmap, get_expmap_der

def print2D(mat, precision=3):
    fmt="%% .%if" % precision
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print fmt % mat[i,j],
        print

class CartesianX(object):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w

    def __repr__(self):
        #return "Cartesian-X %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-X %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            print "Warning: CartesianX same atoms, different weights (%.4f %.4f)" % (self.w, other.w)
        return eq

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
        # return "Cartesian-Y %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-Y %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            print "Warning: CartesianY same atoms, different weights (%.4f %.4f)" % (self.w, other.w)
        return eq

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
        # return "Cartesian-Z %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-Z %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            print "Warning: CartesianZ same atoms, different weights (%.4f %.4f)" % (self.w, other.w)
        return eq

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

class TranslationX(object):
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)

    def __repr__(self):
        # return "Translation-X %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-X %s" % (commadash(self.a))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            print "Warning: TranslationX same atoms, different weights"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,0]*self.w)
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i, a in enumerate(self.a):
            derivatives[a][0] = self.w[i]
        return derivatives

class TranslationY(object):
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)

    def __repr__(self):
        # return "Translation-Y %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-Y %s" % (commadash(self.a))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            print "Warning: TranslationY same atoms, different weights"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,1]*self.w)
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i, a in enumerate(self.a):
            derivatives[a][1] = self.w[i]
        return derivatives

class TranslationZ(object):
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)

    def __repr__(self):
        # return "Translation-Z %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-Z %s" % (commadash(self.a))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            print "Warning: TranslationZ same atoms, different weights"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,2]*self.w)
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i, a in enumerate(self.a):
            derivatives[a][2] = self.w[i]
        return derivatives

class Rotator(object):

    def __init__(self, a, x0):
        self.a = list(tuple(sorted(a)))
        x0 = x0.reshape(-1, 3)
        self.x0 = x0
        self.stored_valxyz = np.zeros_like(x0)
        self.stored_value = None
        self.stored_derxyz = np.zeros_like(x0)
        self.stored_deriv = None
        self.stored_norm = 0.0

    def reset(self, x0):
        x0 = x0.reshape(-1, 3)
        self.x0 = x0
        self.stored_valxyz = np.zeros_like(x0)
        self.stored_value = None
        self.stored_derxyz = np.zeros_like(x0)
        self.stored_deriv = None
        self.stored_norm = 0.0

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.x0-other.x0)**2) > 1e-6:
            print "Warning: Rotator same atoms, different reference positions"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1, 3)
        if np.max(np.abs(xyz-self.stored_valxyz)) < 1e-12:
            return self.stored_value
        else:
            xsel = xyz[self.a, :]
            ysel = self.x0[self.a, :]
            answer = get_expmap(xsel, ysel)
            self.stored_norm = np.linalg.norm(answer)
            # if np.linalg.norm(answer) > 2.5:
            #     print "%s rotation norm is %.3f" % (' '.join([str(i) for i in self.a]), np.linalg.norm(answer))
            self.stored_valxyz = xyz.copy()
            self.stored_value = answer
            return answer

    def derivative(self, xyz):
        xyz = xyz.reshape(-1, 3)
        if np.max(np.abs(xyz-self.stored_derxyz)) < 1e-12:
            return self.stored_deriv
        else:
            xsel = xyz[self.a, :]
            ysel = self.x0[self.a, :]
            deriv_raw = get_expmap_der(xsel, ysel)
            # if len(self.a) > 3: 
            #     print "Rotation derivative:", deriv_raw
            derivatives = np.zeros((xyz.shape[0], 3, 3), dtype=float)
            for i, a in enumerate(self.a):
                derivatives[a, :, :] = deriv_raw[i, :, :]
            self.stored_derxyz = xyz.copy()
            self.stored_deriv = derivatives
            return derivatives

class RotationA(object):
    def __init__(self, a, x0, Rotators, w=1.0):
        self.a = tuple(sorted(a))
        self.x0 = x0
        self.w = w
        if self.a not in Rotators:
            Rotators[self.a] = Rotator(self.a, x0)
        self.Rotator = Rotators[self.a]

    def __repr__(self):
        # return "Rotation-A %s : Weight %.3f" % (' '.join([str(i+1) for i in self.a]), self.w)
        return "Rotation-A %s" % (commadash(self.a))

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        # if eq and np.sum((self.w-other.w)**2) > 1e-6:
        #     print "Warning: RotationA same atoms, different weights"
        # if eq and np.sum((self.x0-other.x0)**2) > 1e-6:
        #     print "Warning: RotationA same atoms, different reference positions"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        return self.Rotator.value(xyz)[0]*self.w
        
    def derivative(self, xyz):
        der_all = self.Rotator.derivative(xyz)
        derivatives = der_all[:, :, 0]*self.w
        return derivatives

class RotationB(object):
    def __init__(self, a, x0, Rotators, w=1.0):
        self.a = tuple(sorted(a))
        self.x0 = x0
        self.w = w
        if self.a not in Rotators:
            Rotators[self.a] = Rotator(self.a, x0)
        self.Rotator = Rotators[self.a]

    def __repr__(self):
        # return "Rotation-B %s : Weight %.3f" % (' '.join([str(i+1) for i in self.a]), self.w)
        return "Rotation-B %s" % (commadash(self.a))

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        # if eq and np.sum((self.w-other.w)**2) > 1e-6:
        #     print "Warning: RotationB same atoms, different weights"
        # if eq and np.sum((self.x0-other.x0)**2) > 1e-6:
        #     print "Warning: RotationB same atoms, different reference positions"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        return self.Rotator.value(xyz)[1]*self.w
        
    def derivative(self, xyz):
        der_all = self.Rotator.derivative(xyz)
        derivatives = der_all[:, :, 1]*self.w
        return derivatives

class RotationC(object):
    def __init__(self, a, x0, Rotators, w=1.0):
        self.a = tuple(sorted(a))
        self.x0 = x0
        self.w = w
        if self.a not in Rotators:
            Rotators[self.a] = Rotator(self.a, x0)
        self.Rotator = Rotators[self.a]

    def __repr__(self):
        # return "Rotation-C %s : Weight %.3f" % (' '.join([str(i+1) for i in self.a]), self.w)
        return "Rotation-C %s" % (commadash(self.a))

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        # if eq and np.sum((self.w-other.w)**2) > 1e-6:
        #     print "Warning: RotationC same atoms, different weights"
        # if eq and np.sum((self.x0-other.x0)**2) > 1e-6:
        #     print "Warning: RotationC same atoms, different reference positions"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        return self.Rotator.value(xyz)[2]*self.w
        
    def derivative(self, xyz):
        der_all = self.Rotator.derivative(xyz)
        derivatives = der_all[:, :, 2]*self.w
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
        return "Out-of-Plane %i-%i-%i-%i" % (self.a+1, self.b+1, self.c+1, self.d+1)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if set([self.b, self.c, self.d]) == set([other.b, other.c, other.d]):
                if [self.b, self.c, self.d] != [other.b, other.c, other.d]:
                    print "Warning: OutOfPlane atoms are the same, ordering is different"
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
    def addConstraint(self, cPrim, cVal):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def haveConstraints(self):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def augmentGH(self, xyz, G, H):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def calcGradProj(self, xyz, gradx):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

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
        try:
            G = self.GMatrix(xyz, u)
            time_G = click()
            U, S, VT = np.linalg.svd(G)
            time_svd = click()
        except:
            print "\x1b[1;91m SVD fails, perturbing coordinates and trying again\x1b[0m"
            xyz = xyz + 1e-2*np.random.random(xyz.shape)
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
            # print "%.5e % .5e" % (ival,value)
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
            print "IC %i/%i : %s" % (i, Analytical.shape[0], self.Internals[i]), 
            lines = [""]
            maxerr = 0.0
            for j in range(Analytical.shape[1]):
                lines.append("Atom %i" % (j+1))
                for k in range(Analytical.shape[2]):
                    lines.append("xyz"[k])
                    error = Analytical[i,j,k] - FiniteDifference[i,j,k]
                    if np.abs(error) > 1e-5:
                        color = "\x1b[91m"
                    else:
                        color = "\x1b[92m"
                    lines.append("% .5e % .5e %s% .5e\x1b[0m" % (Analytical[i,j,k], FiniteDifference[i,j,k], color, Analytical[i,j,k] - FiniteDifference[i,j,k]))
                    if maxerr < np.abs(error):
                        maxerr = np.abs(error)
            if maxerr > 1e-5:
                print '\n'.join(lines)
            else:
                print "Max Error = %.5e" % maxerr
                
        print "Finite-difference Finished"

    def calcGrad(self, xyz, gradx):
        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
        return np.array(Gq).flatten()

    def readCache(self, xyz, dQ):
        if not hasattr(self, 'stored_xyz'):
            return None
        xyz = xyz.flatten()
        dQ = dQ.flatten()
        if np.linalg.norm(self.stored_xyz - xyz) < 1e-10:
            if np.linalg.norm(self.stored_dQ - dQ) < 1e-10:
                return self.stored_newxyz
        return None

    def writeCache(self, xyz, dQ, newxyz):
        xyz = xyz.flatten()
        dQ = dQ.flatten()
        newxyz = newxyz.flatten()
        self.stored_xyz = xyz.copy()
        self.stored_dQ = dQ.copy()
        self.stored_newxyz = newxyz.copy()

    def newCartesian(self, xyz, dQ, u=None, verbose=True, applyCon=True):
        cached = self.readCache(xyz, dQ)
        if cached is not None:
            # print "Returning cached result"
            return cached
        xyz1 = xyz.copy()
        dQ1 = dQ.copy()
        # Iterate until convergence:
        microiter = 0
        ndqs = []
        rmsds = []
        self.bork = False
        # Damping factor
        damp = 1.0
        # Function to exit from loop
        def finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1):
            if ndqt > 1e-1:
                if verbose: print "Failed to obtain coordinates after %i microiterations (rmsd = %.3e |dQ| = %.3e)" % (microiter, rmsdt, ndqt)
                self.bork = True
                self.writeCache(xyz, dQ, xyz_iter1)
                return xyz_iter1.flatten()
            elif ndqt > 1e-3:
                if verbose: print "Approximate coordinates obtained after %i microiterations (rmsd = %.3e |dQ| = %.3e)" % (microiter, rmsdt, ndqt)
            else:
                if verbose: print "Cartesian coordinates obtained after %i microiterations (rmsd = %.3e |dQ| = %.3e)" % (microiter, rmsdt, ndqt)
            # These two lines of code make sure that we remove any residual constraint violations
            # Presently it is broken
            if applyCon:
                print "newCartesian calling applyConstraints"
                xyzCon = self.applyConstraints(xyzsave.flatten())
                xyzsave = xyzCon.copy()
            self.writeCache(xyz, dQ, xyzsave)
            return xyzsave.flatten()
        fail_counter = 0
        while True:
            microiter += 1
            Bmat = np.matrix(self.wilsonB(xyz1))
            Ginv = self.GInverse(xyz1, u)
            # Get new Cartesian coordinates
            if u is not None:
                dxyz = damp*u*Bmat.T*Ginv*(np.matrix(dQ1).T)
            else:
                dxyz = damp*Bmat.T*Ginv*(np.matrix(dQ1).T)
            xyz2 = xyz1 + np.array(dxyz).flatten()
            if microiter == 1:
                xyzsave = xyz2.copy()
                xyz_iter1 = xyz2.copy()
            # Calculate the actual change in internal coordinates
            dQ_actual = self.calcDiff(xyz2, xyz1)
            rmsd = np.sqrt(np.mean((np.array(xyz2-xyz1).flatten())**2))
            ndq = np.linalg.norm(dQ1-dQ_actual)
            if len(ndqs) > 0:
                if ndq > ndqt:
                    if verbose: print "Iter: %i Err-dQ (Best) = %.5e (%.5e) RMSD: %.5e Damp: %.5e (Bad)" % (microiter, ndq, ndqt, rmsd, damp)
                    damp /= 2
                    fail_counter += 1
                    # xyz2 = xyz1.copy()
                else:
                    if verbose: print "Iter: %i Err-dQ (Best) = %.5e (%.5e) RMSD: %.5e Damp: %.5e (Good)" % (microiter, ndq, ndqt, rmsd, damp)
                    fail_counter = 0
                    damp = min(damp*1.2, 1.0)
                    rmsdt = rmsd
                    ndqt = ndq
                    xyzsave = xyz2.copy()
            else:
                if verbose: print "Iter: %i Err-dQ = %.5e RMSD: %.5e Damp: %.5e" % (microiter, ndq, rmsd, damp)
                rmsdt = rmsd
                ndqt = ndq
            ndqs.append(ndq)
            rmsds.append(rmsd)
            # Check convergence / fail criteria
            if rmsd < 1e-6 or ndq < 1e-6:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            if fail_counter >= 5:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            if microiter == 50:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            # Figure out the further change needed
            dQ1 = dQ1 - dQ_actual
            xyz1 = xyz2.copy()
    
class PrimitiveInternalCoordinates(InternalCoordinates):
    def __init__(self, molecule, connect=False):
        self.connect = connect
        self.Internals = []
        self.cPrims = []
        self.cVals = []
        self.Rotators = OrderedDict()
        self.elem = molecule.elem
        if len(molecule) != 1:
            raise RuntimeError('Only one frame allowed in molecule object')
        # Determine the atomic connectivity
        molecule.build_topology()
        frags = [m.nodes() for m in molecule.molecules]
        # coordinates in Angstrom
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
        ThrowCarts = False
        if not connect:
            if ThrowCarts:
                for i in range(molecule.na):
                    self.add(CartesianX(i, w=1.0))
                    self.add(CartesianY(i, w=1.0))
                    self.add(CartesianZ(i, w=1.0))
            else:
                for i in frags:
                    if len(i) >= 3:
                        self.add(TranslationX(i, w=np.ones(len(i))/len(i)))
                        self.add(TranslationY(i, w=np.ones(len(i))/len(i)))
                        self.add(TranslationZ(i, w=np.ones(len(i))/len(i)))
                        # Reference coordinates are given in Bohr.
                        sel = coords.reshape(-1,3)[i,:] / 0.529
                        sel -= np.mean(sel, axis=0)
                        rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
                        self.add(RotationA(i, coords / 0.529, self.Rotators, w=rg))
                        self.add(RotationB(i, coords / 0.529, self.Rotators, w=rg))
                        self.add(RotationC(i, coords / 0.529, self.Rotators, w=rg))
                    else:
                        for j in i:
                            self.add(CartesianX(j, w=1.0))
                            self.add(CartesianY(j, w=1.0))
                            self.add(CartesianZ(j, w=1.0))

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
            self.add(Distance(a, b))

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
                            self.add(Angle(a, b, c))
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
                            carts = [CartesianX, CartesianY, CartesianZ]
                            # Add two of the most perpendicular Cartesian coordinates
                            for i in np.argsort(dots)[:2]:
                                self.add(carts[i](b))

        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    for d in molecule.topology.neighbors(b):
                        if a < c < d:
                            nnc = (min(a, b), max(a, b)) in noncov
                            nnc += (min(b, c), max(b, c)) in noncov
                            nnc += (min(b, d), max(b, d)) in noncov
                            # if nnc >= 1: continue
                            for i, j, k in sorted(list(itertools.permutations([a, c, d], 3))):
                                Ang1 = Angle(b,i,j)
                                Ang2 = Angle(i,j,k)
                                if np.abs(np.cos(Ang1.value(coords))) > LinThre: continue
                                if np.abs(np.cos(Ang2.value(coords))) > LinThre: continue
                                if np.abs(np.dot(Ang1.normal_vector(coords), Ang2.normal_vector(coords))) > 0.95:
                                    self.delete(Angle(i, b, j))
                                    self.add(OutOfPlane(b, i, j, k))
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
                        if b < c:
                            self.add(Dihedral(a, b, c, d))
                        else:
                            self.add(Dihedral(d, c, b, a))

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

    def __ne__(self, other):
        return not self.__eq__(other)

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

    def resetRotations(self, xyz):
        for rot in self.Rotators.values():
            rot.reset(xyz)

    def largeRots(self):
        for Internal in self.Internals:
            if type(Internal) in [RotationA, RotationB, RotationC]:
                if Internal.Rotator.stored_norm > 0.9*np.pi:
                    return True
        return False

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

    def add(self, dof):
        if dof not in self.Internals:
            self.Internals.append(dof)

    def delete(self, dof):
        for ii in range(len(self.Internals))[::-1]:
            if dof == self.Internals[ii]:
                del self.Internals[ii]

    def addConstraint(self, cPrim, cVal=None, xyz=None):
        if cVal is None and xyz is None:
            raise RuntimeError('Please provide either cval or xyz')
        if cVal is None:
            # If coordinates are provided instead of a constraint value, 
            # then calculate the constraint value from the positions.
            # If both are provided, then the coordinates are ignored.
            cVal = cPrim.value(xyz)
        if cPrim in self.cPrims:
            print "Updating constraint value to %.4e" % (cVal)
            iPrim = self.cPrims.index(cPrim)
            self.cVals[iPrim] = cVal
        if cPrim not in self.Internals:
            self.Internals.append(cPrim)
        self.cPrims.append(cPrim)
        self.cVals.append(cVal)

    def getConstraint_from(self, other):
        if other.haveConstraints():
            for cPrim, cVal in zip(other.cPrims, other.cVals):
                self.addConstraint(cPrim, cVal)

    def haveConstraints(self):
        return len(self.cPrims) > 0

    def printConstraints(self, xyz):
        nc = len(self.cPrims)
        print "Constraint                         Current      Target       Diff."
        for ic, c in enumerate(self.cPrims):
            current = c.value(xyz)
            reference = self.cVals[ic]
            diff = (current - reference)
            if np.abs(diff-2*np.pi) < np.abs(diff):
                diff -= 2*np.pi
            if np.abs(diff+2*np.pi) < np.abs(diff):
                diff += 2*np.pi
            if type(c) in [TranslationX, TranslationY, TranslationZ, CartesianX, CartesianY, CartesianZ, Distance]:
                factor = 0.529
            elif type(c) in [Angle, Dihedral, OutOfPlane]:
                factor = 180.0/np.pi
            print "%-30s  % 10.5f  % 10.5f  % 10.5f" % (str(c), current*factor, reference*factor, diff*factor)

#     def augmentGH(self, xyz, G, H):
#         # Number of internals (elements of G)
#         ni = len(G)
#         # Number of constraints
#         nc = len(self.cPrims)
#         # Total dimension
#         nt = ni+nc
#         # Lower block of the augmented Hessian
#         cT = np.zeros((nc, ni), dtype=float)
#         c0 = np.zeros(nc, dtype=float)
#         for ic, c in enumerate(self.cPrims):
#             # Look up the index of the primitive that is being constrained
#             iPrim = self.Internals.index(c)
#             # The constraint indexed by iC is simply the primitive indexed by iPrim
#             cT[ic, iPrim] = 1.0
#             # Calculate the further change needed in this constrained variable
#             c0[ic] = self.cVals[ic] - c.value(xyz)
#         # Construct augmented Hessian
#         HC = np.zeros((nt, nt), dtype=float)
#         HC[0:ni, 0:ni] = H[:,:]
#         HC[ni:nt, 0:ni] = cT[:,:]
#         HC[0:ni, ni:nt] = cT.T[:,:]
#         # Construct augmented gradient
#         GC = np.zeros(nt, dtype=float)
#         GC[0:ni] = G[:]
#         GC[ni:nt] = -c0[:]
#         return HC, GC
# 
#     def calcGradProj(self, xyz, gradx):
#         if len(self.cPrims) == 0:
#             return gradx
#         q0 = self.calculate(xyz)
#         Ginv = self.GInverse(xyz)
#         Bmat = self.wilsonB(xyz)
#         # Internal coordinate gradient
#         Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
#         Gqc = np.array(Gq).flatten()
#         for ic, c in enumerate(self.cPrims):
#             # Look up the index of the primitive that is being constrained
#             iPrim = self.Internals.index(c)
#             # Zero out that component of the gradient
#             Gq[iPrim] = 0.0
#         Gxc = np.array(np.matrix(Bmat.T)*np.matrix(Gqc).T).flatten()
#         return Gxc

    # def applyConstraints(self, xyz):
    #     # # Number of internals (elements of G)
    #     # ni = len(G)
    #     # # Number of constraints
    #     # nc = len(self.cPrims)
    #     # # Total dimension
    #     # nt = ni+nc
    #     # # Lower block of the augmented Hessian
    #     # cT = np.zeros((nc, ni), dtype=float)
    #     # c0 = np.zeros(nc, dtype=float)
    #     dQ = np.zeros(len(self.Internals), dtype=float)
    #     for ic, c in enumerate(self.cPrims):
    #         # Look up the index of the primitive that is being constrained
    #         iPrim = self.Internals.index(c)
    #         # # The constraint indexed by iC is simply the primitive indexed by iPrim
    #         # cT[ic, iPrim] = 1.0
    #         # Calculate the further change needed in this constrained variable
    #         dQ[iPrim] = self.cVals[ic] - c.value(xyz)
    #         if type(c) in [Angle, Dihedral, OutOfPlane]:
    #             Plus2Pi = dQ[iPrim] + 2*np.pi
    #             Minus2Pi = dQ[iPrim] - 2*np.pi
    #             if np.abs(dQ[iPrim]) > np.abs(Plus2Pi):
    #                 dQ[iPrim] = Plus2Pi
    #             if np.abs(dQ[iPrim]) > np.abs(Minus2Pi):
    #                 dQ[iPrim] = Minus2Pi
    #     print dQ
    #     return self.newCartesian(xyz, dQ, verbose=True)
        # # Construct augmented Hessian
        # HC = np.zeros((nt, nt), dtype=float)
        # HC[0:ni, 0:ni] = H[:,:]
        # HC[ni:nt, 0:ni] = cT[:,:]
        # HC[0:ni, ni:nt] = cT.T[:,:]
        # # Construct augmented gradient
        # GC = np.zeros(nt, dtype=float)
        # GC[0:ni] = G[:]
        # GC[ni:nt] = -c0[:]
        # return HC, GC
        
    
    def guess_hessian(self, coords):
        xyzs = coords.reshape(-1,3)*0.529
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
                Hdiag.append(0.05)
            elif type(ic) in [TranslationX, TranslationY, TranslationZ]:
                Hdiag.append(0.05)
            elif type(ic) in [RotationA, RotationB, RotationC]:
                Hdiag.append(0.05)
            else:
                raise RuntimeError('Spoo!')
        return np.matrix(np.diag(Hdiag))

class CartesianCoordinates(InternalCoordinates):
    def __init__(self, molecule, build=False, connect=False):
        self.Internals = []
        self.elem = molecule.elem
        if len(molecule) != 1:
            raise RuntimeError('Only one frame allowed in molecule object')
        for i in range(molecule.na):
            self.add(CartesianX(i, w=1.0))
            self.add(CartesianY(i, w=1.0))
            self.add(CartesianZ(i, w=1.0))

    def __eq__(self, other):
        answer = True
        for i in self.Internals:
            if i not in other.Internals:
                answer = False
        for i in other.Internals:
            if i not in self.Internals:
                answer = False
        return answer

    def __ne__(self, other):
        return not self.__eq__(other)

    def calcDiff(self, coord1, coord2):
        """ Calculate difference in internal coordinates, accounting for changes in 2*pi of angles. """
        Q1 = self.calculate(coord1)
        Q2 = self.calculate(coord2)
        PMDiff = (Q1-Q2)
        return PMDiff

    def calculate(self, xyz):
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.value(xyz))
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

    def GInverse(self, xyz, u=None):
        return self.GInverse_SVD(xyz, u)

    def guess_hessian(self, coords):
        return np.eye(len(self.Internals)) * 0.05

class DelocalizedInternalCoordinates(InternalCoordinates):
    def __init__(self, molecule, build=False, connect=False, constraints=None, cvals=None):
        self.Prims = PrimitiveInternalCoordinates(molecule, connect)
        self.connect = connect
        xyz = molecule.xyzs[0].flatten() / 0.529
        self.na = molecule.na
        if constraints is not None:
            if len(constraints) != len(cvals):
                raise RuntimeError("List of constraints should be same length as constraint values")
            for cons, cval in zip(constraints, cvals):
                self.addConstraint(cons, cval, xyz)
        if build:
            self.build_dlc(xyz)

    def __repr__(self):
        return self.Prims.__repr__()
            
    def update(self, other):
        return self.Prims.update(other.Prims)

    def addConstraint(self, cPrim, cVal, xyz):
        """
        Add a constrained degree of freedom.
        """
        self.Prims.addConstraint(cPrim, cVal, xyz)

    def getConstraints_from(self, other):
        self.Prims.getConstraint_from(other.Prims)
        
    def haveConstraints(self):
        return len(self.Prims.cPrims) > 0

    def printConstraints(self, xyz):
        self.Prims.printConstraints(xyz)

    def augmentGH(self, xyz, G, H):
        # Number of internals (elements of G)
        ni = len(G)
        # Number of constraints
        nc = len(self.Prims.cPrims)
        # Augmented versions of gradient and Hessian
        GC = G.copy()
        HC = H.copy()
        for ic, c in enumerate(self.Prims.cPrims):
            # Look up the index of the primitive that is being constrained
            iPrim = self.Prims.Internals.index(c)
            nr = self.cDLC[ic]
            # print nr
            HC[nr, :] = 0.0
            HC[:, nr] = 0.0
            HC[nr, nr] = 1.0
            # Calculate the further change needed in this constrained variable
            GC[nr] = -(self.Prims.cVals[ic] - c.value(xyz))
            if type(c) in [Angle, Dihedral, OutOfPlane]:
                Plus2Pi = GC[nr] + 2*np.pi
                Minus2Pi = GC[nr] - 2*np.pi
                if np.abs(GC[nr]) > np.abs(Plus2Pi):
                    GC[nr] = Plus2Pi
                if np.abs(GC[nr]) > np.abs(Minus2Pi):
                    GC[nr] = Minus2Pi
            GC[nr] = 0.0
        return HC, GC

    # # This is the one that wurx
    # def augmentGH(self, xyz, G, H):
    #     # Number of internals (elements of G)
    #     ni = len(G)
    #     # Number of constraints
    #     nc = len(self.Prims.cPrims)
    #     # Total dimension
    #     nt = ni+nc
    #     # Lower block of the augmented Hessian
    #     cT = np.zeros((nc, ni), dtype=float)
    #     c0 = np.zeros(nc, dtype=float)
    #     for ic, c in enumerate(self.Prims.cPrims):
    #         # Look up the index of the primitive that is being constrained
    #         iPrim = self.Prims.Internals.index(c)
    #         # Get the corresponding linear combination coefficients from the
    #         # DLC that allows us to calculate the primitive
    #         # Note that the constraint DLCs occur at the end, and in increasing order
    #         cT[ic, ni-nc+ic] = 1.0
    #         # cT[ic, :] = np.array(self.Vecs[iPrim,:]).flatten()
    #         # Calculate the further change needed in this constrained variable
    #         c0[ic] = self.Prims.cVals[ic] - c.value(xyz)
    #         if type(c) in [Angle, Dihedral, OutOfPlane]:
    #             Plus2Pi = c0[ic] + 2*np.pi
    #             Minus2Pi = c0[ic] - 2*np.pi
    #             if np.abs(c0[ic]) > np.abs(Plus2Pi):
    #                 c0[ic] = Plus2Pi
    #             if np.abs(c0[ic]) > np.abs(Minus2Pi):
    #                 c0[ic] = Minus2Pi
    #     # Construct augmented Hessian
    #     HC = np.zeros((nt, nt), dtype=float)
    #     HC[0:ni, 0:ni] = H[:,:]
    #     HC[ni:nt, 0:ni] = cT[:,:]
    #     HC[0:ni, ni:nt] = cT.T[:,:]
    #     # print HC
    #     # Construct augmented gradient
    #     GC = np.zeros(nt, dtype=float)
    #     GC[0:ni] = G[:]
    #     GC[ni:nt] = -c0[:]
    #     return HC, GC
    
    def applyConstraints(self, xyz):
        xyz1 = xyz.copy()
        niter = 0
        while True:
            dQ = np.zeros(len(self.Internals), dtype=float)
            for ic, c in enumerate(self.Prims.cPrims):
                # Look up the index of the primitive that is being constrained
                iPrim = self.Prims.Internals.index(c)
                # Look up the index of the DLC that corresponds to the constraint
                iDLC = self.cDLC[ic]
                # Get the corresponding linear combination coefficients from the
                # DLC that allows us to calculate the primitive
                # cT[ic, :] = np.array(self.Vecs[iPrim,:]).flatten()
                # Calculate the further change needed in this constrained variable
                dQ[iDLC] = (self.Prims.cVals[ic] - c.value(xyz1))/self.Vecs[iPrim, iDLC]
                if type(c) in [Angle, Dihedral, OutOfPlane]:
                    Plus2Pi = dQ[iDLC] + 2*np.pi
                    Minus2Pi = dQ[iDLC] - 2*np.pi
                    if np.abs(dQ[iDLC]) > np.abs(Plus2Pi):
                        dQ[iDLC] = Plus2Pi
                    if np.abs(dQ[iDLC]) > np.abs(Minus2Pi):
                        dQ[iDLC] = Minus2Pi
            print "applyConstraints calling newCartesian (%i), |dQ| = %.3e" % (niter, np.linalg.norm(dQ))
            xyz2 = self.newCartesian(xyz1, dQ, verbose=True, applyCon=False)
            if np.linalg.norm(dQ) < 1e-6:
                return xyz2
            if niter > 1 and np.linalg.norm(dQ) > np.linalg.norm(dQ0):
                print "\x1b[1;93mWarning: Failed to apply Constraint\x1b[0m"
                return xyz1
            xyz1 = xyz2.copy()
            niter += 1
            dQ0 = dQ.copy()
            # if niter == 50:
            #     raise RuntimeError("Unable to obtain desired constrained geometry in 50 outer loops")
        # return xyz2
            # print c.value(xyz1)
            # print c.value(xyz2)
            # print dQ
            # raw_input()
            # xyz1 = xyz2.copy()
            # dQ = c.value(xyz
        # import sys
        # sys.exit()
            # dp = c.value(xyz) - np.dot(self.calculate(xyz), cT[ic, :])
                    
        
    # def applyConstraints(self, xyz):
    #     # # return self.Prims.applyConstraints(xyz)
    #     nc = len(self.Prims.cPrims)
    #     ni = len(self.Internals)
    #     # Lower block of the augmented Hessian
    #     cT = np.zeros((nc, ni), dtype=float)
    #     c0 = np.zeros(nc, dtype=float)
    #     # Initial values of the primitive internal coordinates
    #     p0 = self.Prims.calculate(xyz)
    #     dQ = np.zeros(len(self.Prims.Internals), dtype=float)
    #     for ic, c in enumerate(self.Prims.cPrims):
    #         # Look up the index of the primitive that is being constrained
    #         iPrim = self.Prims.Internals.index(c)
    #         # Get the corresponding linear combination coefficients from the
    #         # DLC that allows us to calculate the primitive
    #         cT[ic, :] = np.array(self.Vecs[iPrim,:]).flatten()
    #         # Calculate the further change needed in this constrained variable
    #         c0[ic] = self.Prims.cVals[ic] - c.value(xyz)
    #         if type(c) in [Angle, Dihedral, OutOfPlane]:
    #             Plus2Pi = c0[ic] + 2*np.pi
    #             Minus2Pi = c0[ic] - 2*np.pi
    #             if np.abs(c0[ic]) > np.abs(Plus2Pi):
    #                 c0[ci] = Plus2Pi
    #             if np.abs(c0[ic]) > np.abs(Minus2Pi):
    #                 c0[ic] = Minus2Pi
    #         dp = c.value(xyz) - np.dot(self.calculate(xyz), cT[ic, :])
    #         # print c.value(xyz), 
    #     print self.Vecs.shape
            
        # print c0
        # dQ = flat(np.matrix(cT.T)*col(c0))
        # print dQ
        # xyz1 = self.newCartesian(xyz, dQ, verbose=True)
        # print self.Prims.calculate(xyz1) - self.Prims.calculate(xyz)
    
    def calcGradProj(self, xyz, gradx):
        if len(self.Prims.cPrims) == 0:
            return gradx
        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
        # Schmidt orthogonalization procedure;
        # build a list of mutually orthogonal directions
        # representing the degrees of freedom that we constrained.
        U = []
        for ic, c in enumerate(self.Prims.cPrims):
            # Look up the index of the primitive that is being constrained
            iPrim = self.Prims.Internals.index(c)
            # Linear combination of the DLC that represents the primitive
            pvec = np.array(self.Vecs[iPrim,:]).flatten()
            U.append(pvec.copy())
            for Ui in U[:-1]:
                # Subtract the projection of pvec along Ui
                Uih = Ui / np.linalg.norm(Ui)
                U[-1] -= Uih * np.dot(pvec, Uih)
        # At the end of this loop, we should have a set of mutually orthogonal U vectors
        Gqc = np.array(Gq).flatten()
        for Ui in U:
            # Calculate the projection of pvec along Ui
            Uih = Ui / np.linalg.norm(Ui)
            Gqc -= Uih * np.dot(Gqc, Uih)
        Gxc = np.array(np.matrix(Bmat.T)*np.matrix(Gqc).T).flatten()
        return Gxc

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
            # print ival, value
            if np.abs(value) > 1e-6:
                LargeVals += 1
                LargeIdx.append(ival)
        Expect = 3*self.na - 6
        if self.na == 2:
            Expect = 1
        if self.na == 1:
            Expect = 0
        # print "%i atoms (expect %i coordinates); %i/%i singular values are > 1e-6" % (self.na, Expect, LargeVals, len(L))
        # if LargeVals <= Expect:
        self.Vecs = Q[:, LargeIdx]
        self.Internals = ["DLC %i" % (i+1) for i in range(len(LargeIdx))]

        # Vecs has number of rows equal to the number of primitives, and
        # number of columns equal to the number of delocalized internal coordinates.
        # In the following, assume that we have 72 DLCs, 115 internals and 2 constraints.
        if self.haveConstraints():
            self.cDLC = []
            uSpace = np.eye(self.Vecs.shape[1], dtype=float)
            click()
            print "Projecting out constraints...",
            for ic, c in enumerate(self.Prims.cPrims):
                print ic+1,
                # Look up the index of the primitive that is being constrained
                iPrim = self.Prims.Internals.index(c)
                # Pick a row out of the eigenvector space. This is a linear combination of the DLCs.
                cVec = self.Vecs[iPrim, :]
                cVec = np.array(cVec)
                cVec /= np.linalg.norm(cVec)
                # This is a "new DLC" that corresponds to the primitive that we are constraining
                cProj = self.Vecs*cVec.T
                cProj /= np.linalg.norm(cProj)
                # Now we project the new DLC out of the rest
                U = [np.array(cProj).flatten()]
                # We loop through the DLCs but omit the last one
                n_orth = 0
                for iv in range(self.Vecs.shape[1]):
                    if iv == self.Vecs.shape[1] - 1 + n_orth: break
                    # Pick a column out of the eigenvector space.
                    v = np.array(self.Vecs[:, iv]).flatten()
                    U.append(v.copy())
                    for ui in U[:-1]:
                        U[-1] -= ui * np.dot(ui, v)
                    if np.linalg.norm(U[-1]) < 1e-6: 
                        # This catches the corner case that one of the eigenvectors is already the one we're constraining,
                        # in which case the orthogonal projection will have a norm of zero. In this case, then we skip over
                        # this vector and go to the next one, but this should only happen once.
                        n_orth += 1
                        U = U[:-1]
                        if n_orth > 1:
                            raise RuntimeError("Gram-Schmidt orthogonalization has failed")
                        continue
                    U[-1] /= np.linalg.norm(U[-1])
                self.Vecs = np.matrix(U).T
            print "Done in", click(), "s"
            # Rearrange the DLCs so that the ones corresponding to constraints are last, and in the same order that they were entered
            self.Vecs = np.hstack((self.Vecs[:, len(self.Prims.cPrims):], self.Vecs[:, :len(self.Prims.cPrims)][:, ::-1]))
            self.cDLC = [self.Vecs.shape[1] - len(self.Prims.cPrims) + i for i in range(len(self.Prims.cPrims))]

    def weight_vectors(self, xyz):
        Bmat = np.matrix(self.wilsonB(xyz))
        Ginv = self.GInverse(xyz, None)
        eps = 1e-6
        dxdq = np.zeros(len(self.Internals))
        for i in range(len(self.Internals)):
            dQ = np.zeros(len(self.Internals), dtype=float)
            dQ[i] = eps
            dxyz = Bmat.T*Ginv*(np.matrix(dQ).T)
            rmsd = np.sqrt(np.mean(np.sum(np.array(dxyz).reshape(-1,3)**2, axis=1)))
            # rmsd = dxyz**2
            dxdq[i] = rmsd/eps
        dxdq /= np.min(dxdq)
        for i in range(len(self.Internals)):
            self.Vecs[:, i] *= dxdq[i]
        #print dxdq

    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)

    def largeRots(self):
        return self.Prims.largeRots()

    def calcDiff(self, coord1, coord2):
        """ Calculate difference in internal coordinates, accounting for changes in 2*pi of angles. """
        PMDiff = self.Prims.calcDiff(coord1, coord2)
        Answer = np.matrix(PMDiff)*self.Vecs
        return np.array(Answer).flatten()

    def calculate(self, coords):
        PrimVals = self.Prims.calculate(coords)
        Answer = np.matrix(PrimVals)*self.Vecs
        # To obtain the primitive coordinates from the delocalized internal coordinates,
        # simply multiply self.Vecs*Answer.T where Answer.T is the column vector of delocalized
        # internal coordinates. That means the "c's" in Equation 23 of Schlegel's review paper
        # are simply the rows of the Vecs matrix.
        # print np.dot(np.array(self.Vecs[0,:]).flatten(), np.array(Answer).flatten())
        # print PrimVals[0]
        # raw_input()
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
        return self.GInverse_SVD(xyz, u)

    def repr_diff(self, other):
        return self.Prims.repr_diff(other.Prims)

    def guess_hessian(self, coords):
        Hprim = np.matrix(self.Prims.guess_hessian(coords))
        return np.array(self.Vecs.T*Hprim*self.Vecs)
        # Vi = np.matrix(np.linalg.pinv(self.Vecs))
        # cbh = np.array(Vi*Hprim*Vi.T)
        # return cbh

    def resetRotations(self, xyz):
        self.Prims.resetRotations(xyz)
