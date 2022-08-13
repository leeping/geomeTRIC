"""
internal.py: Internal coordinate systems

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

import itertools
import time, sys
from collections import OrderedDict, defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
from numpy.linalg import multi_dot

from geometric.molecule import Molecule, Elements, Radii
from geometric.nifty import click, commadash, ang2bohr, bohr2ang, logger, pvec1d, pmat2d
from geometric.rotate import get_expmap, get_expmap_der, calc_rot_vec_diff, get_quat, build_F, sorted_eigh

## Some vector calculus functions
def unit_vector(a):
    """
    Vector function: Given a vector a, return the unit vector
    """
    return a / np.linalg.norm(a)

def d_unit_vector(a, ndim=3):
    term1 = np.eye(ndim)/np.linalg.norm(a)
    term2 = np.outer(a, a)/(np.linalg.norm(a)**3)
    answer = term1-term2
    return answer

def d_cross(a, b):
    """
    Given two vectors a and b, return the gradient of the cross product axb w/r.t. a.
    (Note that the answer is independent of a.)
    Derivative is on the first axis.
    """
    d_cross = np.zeros((3, 3), dtype=float)
    for i in range(3):
        ei = np.zeros(3, dtype=float)
        ei[i] = 1.0
        d_cross[i] = np.cross(ei, b)
    return d_cross

def d_cross_ab(a, b, da, db):
    """
    Given two vectors a, b and their derivatives w/r.t. a parameter, return the derivative
    of the cross product
    """
    answer = np.zeros((da.shape[0], 3), dtype=float)
    for i in range(da.shape[0]):
        answer[i] = np.cross(a, db[i]) + np.cross(da[i], b)
    return answer

def ncross(a, b):
    """
    Scalar function: Given vectors a and b, return the norm of the cross product
    """
    cross = np.cross(a, b)
    return np.linalg.norm(cross)

def d_ncross(a, b):
    """
    Return the gradient of the norm of the cross product w/r.t. a
    """
    ncross = np.linalg.norm(np.cross(a, b))
    term1 = a * np.dot(b, b)
    term2 = -b * np.dot(a, b)
    answer = (term1+term2)/ncross
    return answer

def nudot(a, b):
    r"""
    Given two vectors a and b, return the dot product (\hat{a}).b.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(ev, b)
    
def d_nudot(a, b):
    r"""
    Given two vectors a and b, return the gradient of 
    the norm of the dot product (\hat{a}).b w/r.t. a.
    """
    return np.dot(d_unit_vector(a), b)

def ucross(a, b):
    r"""
    Given two vectors a and b, return the cross product (\hat{a})xb.
    """
    ev = a / np.linalg.norm(a)
    return np.cross(ev, b)
    
def d_ucross(a, b):
    r"""
    Given two vectors a and b, return the gradient of 
    the cross product (\hat{a})xb w/r.t. a.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(d_unit_vector(a), d_cross(ev, b))

def nucross(a, b):
    r"""
    Given two vectors a and b, return the norm of the cross product (\hat{a})xb.
    """
    ev = a / np.linalg.norm(a)
    return np.linalg.norm(np.cross(ev, b))
    
def d_nucross(a, b):
    r"""
    Given two vectors a and b, return the gradient of 
    the norm of the cross product (\hat{a})xb w/r.t. a.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(d_unit_vector(a), d_ncross(ev, b))
## End vector calculus functions

class PrimitiveCoordinate(object):
    """
    Parent class for primitive internal coordinate objects with common methods.
    """
    def calcDiff(self, xyz1, xyz2=None, val2=None):
        """
        Return the difference of the internal coordinate
        calculated for c(xyz1) - c(xyz2) or c(xyz1) - val2.

        Parameters
        ----------
        xyz1 : np.ndarray
            xyz coordinates of first structure in Bohr
        xyz2 : np.ndarray
            If provided, xyz coordinates of second structure in Bohr
        val2 : float
            If provided, this is the value to subtract
        """
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        if xyz2 is not None:
            val2 = self.value(xyz2)
        diff = self.value(xyz1) - val2
        if hasattr(self, 'w'):
            w = self.w
        else:
            w = 1.0
        # Divide by the weight, if exists, to get the "base" number
        diff /= w
        # Subtract out any differences of 2*pi for periodic degrees of freedom
        # (rotation ICs handled separately)
        if hasattr(self, 'isPeriodic') and self.isPeriodic:
            Plus2Pi = diff + 2*np.pi
            Minus2Pi = diff - 2*np.pi
            if np.abs(diff) > np.abs(Plus2Pi):
                diff = Plus2Pi
            if np.abs(diff) > np.abs(Minus2Pi):
                diff = Minus2Pi
        diff *= w
        return diff
        
class CartesianX(PrimitiveCoordinate):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        #return "Cartesian-X %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-X %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            logger.warning("Warning: CartesianX same atoms, different weights (%.4f %.4f)\n" % (self.w, other.w))
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

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2
    
class CartesianY(PrimitiveCoordinate):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Cartesian-Y %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-Y %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            logger.warning("Warning: CartesianY same atoms, different weights (%.4f %.4f)\n" % (self.w, other.w))
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

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class CartesianZ(PrimitiveCoordinate):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Cartesian-Z %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-Z %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            logger.warning("Warning: CartesianZ same atoms, different weights (%.4f %.4f)\n" % (self.w, other.w))
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

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2
    
class TranslationX(PrimitiveCoordinate):
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Translation-X %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-X %s" % (commadash(self.a))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            logger.warning("Warning: TranslationX same atoms, different weights\n")
            eq = False
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,0]*self.w)

    def calcDiff(self, xyz1, xyz2=None, val2=None):
        # Translation ICs require an explicit implementation of calcDiff
        # because self.w is not a float but an array
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        if xyz2 is not None:
            val2 = self.value(xyz2)
        diff = self.value(xyz1) - val2
        return diff

    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i, a in enumerate(self.a):
            derivatives[a][0] = self.w[i]
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2
    
class TranslationY(object):
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Translation-Y %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-Y %s" % (commadash(self.a))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            logger.warning("Warning: TranslationY same atoms, different weights\n")
            eq = False
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,1]*self.w)

    def calcDiff(self, xyz1, xyz2=None, val2=None):
        # Translation ICs require an explicit implementation of calcDiff
        # because self.w is not a float but an array
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        if xyz2 is not None:
            val2 = self.value(xyz2)
        diff = self.value(xyz1) - val2
        return diff
    
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i, a in enumerate(self.a):
            derivatives[a][1] = self.w[i]
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class TranslationZ(PrimitiveCoordinate):
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Translation-Z %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-Z %s" % (commadash(self.a))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            logger.warning("Warning: TranslationZ same atoms, different weights\n")
            eq = False
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,2]*self.w)
        
    def calcDiff(self, xyz1, xyz2=None, val2=None):
        # Translation ICs require an explicit implementation of calcDiff
        # because self.w is not a float but an array
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        if xyz2 is not None:
            val2 = self.value(xyz2)
        diff = self.value(xyz1) - val2
        return diff
    
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i, a in enumerate(self.a):
            derivatives[a][2] = self.w[i]
        return derivatives
    
    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class Rotator(object):

    def __init__(self, a, x0):
        self.a = list(tuple(sorted(a)))
        x0 = x0.reshape(-1, 3)
        self.x0 = x0.copy()
        # Information about the regularization quaternion
        self.rquat = np.array([1.0, 0.0, 0.0, 0.0])
        self.rmode = 0
        self.set_regularization(x0)
        self.clear_cache(x0)

    def clear_cache(self, xyz):
        xyz = xyz.reshape(-1, 3)
        self.stored_valxyz = np.zeros_like(xyz)
        self.stored_value = None
        # A second set of xyz coordinates used only when computing
        # differences in rotation coordinates
        self.stored_valxyz2 = np.zeros_like(xyz)
        self.stored_value2 = None
        self.stored_derxyz = np.zeros_like(xyz)
        self.stored_deriv = None
        self.stored_deriv2xyz = np.zeros_like(xyz)
        self.stored_deriv2 = None
        self.stored_norm = 0.0

    def reset(self, x0):
        x0 = x0.reshape(-1, 3)
        self.x0 = x0.copy()
        self.rquat = np.array([1.0, 0.0, 0.0, 0.0])
        self.rmode = 0
        self.set_regularization(x0)
        self.clear_cache(x0)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.x0-other.x0)**2) > 1e-6:
            logger.warning("Warning: Rotator same atoms, different reference positions\n")
        return eq

    def __repr__(self):
        return "Rotator %s" % commadash(self.a)

    def __ne__(self, other):
        return not self.__eq__(other)

    def set_regularization(self, xyz):
        self.clear_cache(xyz)
        x = xyz.reshape(-1, 3)[self.a, :].copy()
        x = x - np.mean(x,axis=0)
        L, Q = sorted_eigh(build_F(x, x))
        # The switch for the regularization is 'sticky'.
        thre_lo = 1.01
        thre_mid = 1.03
        thre_hi = 1.1
        regularization_changed = False
        if L[0]/L[1] < thre_lo and L[0]/L[1] > 0.0 and self.rmode in (-1, 0):
            self.rnorm = 1e-1*L[0]
            self.rmode = 1
            # Uncomment the below four logging messages to check whether regularization is on or off
            # logger.info(" >>> %-18s L[0] = %.3f, L[0]/L[1] = %.3f (linear), turning regularization on.\n" % (str(self), L[0], L[0]/L[1]))
            regularization_changed = True
        elif L[0]/L[1] > thre_hi and self.rmode in (1, 0):
            self.rnorm = 0.0
            self.rmode = -1
            # logger.info(" >>> %s L[0] = %.3f, L[0]/L[1] = %.3f (nonlin), turning regularization off.\n" % (str(self), L[0], L[0]/L[1]))
            regularization_changed = True
        elif self.rmode == 0:
            if L[0]/L[1] < thre_mid and L[0]/L[1] > 0.0:
                # logger.info(" >>> %s L[0] = %.3f, L[0]/L[1] = %.3f (linear), turning regularization on.\n" % (str(self), L[0], L[0]/L[1]))
                self.rnorm = 1e-1*L[0]
                self.rmode = 1
                regularization_changed = True
            else:
                # 
                # logger.info(" >>> %s L[0] = %.3f, L[0]/L[1] = %.3f (nonlin), turning regularization off.\n" % (str(self), L[0], L[0]/L[1]))
                self.rnorm = 0.0
                self.rmode = -1
                regularization_changed = True

        if self.rmode > 0:
            y = self.x0[self.a, :].copy()
            y = y - np.mean(y,axis=0)
            q = get_quat(x, y, r=self.rnorm*self.rquat)
            ang = 2*np.arccos(np.dot(self.rquat, q))
            # self.qsav = q.copy()
            if ang > 0.9*np.pi:
                logger.info("%s angle %.3f\n" % (str(self), ang))
                # logger.info("setting regularization quaternion to %s\n", str(q))
                # self.rquat = q.copy()
        return regularization_changed
        # else:
        #     self.rquat = np.array([1.0, 0.0, 0.0, 0.0])

    def value(self, xyz, store=True):
        xyz = xyz.reshape(-1, 3)
        if np.max(np.abs(xyz-self.stored_valxyz)) < 1e-12:
            return self.stored_value
        else:
            xsel = xyz[self.a, :]
            ysel = self.x0[self.a, :]
            answer = get_expmap(xsel, ysel, r=self.rnorm*self.rquat)
            if store:
                self.stored_norm = np.linalg.norm(answer)
                self.stored_valxyz = xyz.copy()
                self.stored_value = answer.copy()
            return answer

    def visualize(self, xyz):
        xyz = xyz.reshape(-1, 3)
        xsel = xyz[self.a, :]
        ysel = self.x0[self.a, :]
        xmean = np.mean(xsel,axis=0)
        ymean = np.mean(ysel,axis=0)
        answer = np.zeros((2, 3), dtype=float)
        answer[0, :] = xmean
        answer[1, :] = xmean + get_expmap(xsel, ysel, r=self.rnorm*self.rquat)*ang2bohr
        return answer

    def calcDiff(self, xyz1, xyz2=None, val2=None):
        """
        Return the difference of the internal coordinate
        calculated for (xyz1 - xyz2).
        """
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        val1 = self.value(xyz1)
        if xyz2 is not None:
            # The "second" coordinate set is cached separately
            xyz2 = xyz2.reshape(-1, 3)
            if np.max(np.abs(xyz2-self.stored_valxyz2)) < 1e-12:
                val2 = self.stored_value2.copy()
            else:
                val2 = self.value(xyz2, store=False)
                self.stored_valxyz2 = xyz2.copy()
                self.stored_value2 = val2.copy()
        # Calculate difference in rotation vectors, modulo n*2pi displacement vectors
        return calc_rot_vec_diff(val1, val2)

    def derivative(self, xyz):
        xyz = xyz.reshape(-1, 3)
        if np.max(np.abs(xyz-self.stored_derxyz)) < 1e-12:
            return self.stored_deriv
        else:
            xsel = xyz[self.a, :]
            ysel = self.x0[self.a, :]
            xmean = np.mean(xsel,axis=0)
            ymean = np.mean(ysel,axis=0)
            deriv_raw = get_expmap_der(xsel, ysel, r=self.rnorm*self.rquat)
            derivatives = np.zeros((xyz.shape[0], 3, 3), dtype=float)
            for i, a in enumerate(self.a):
                derivatives[a, :, :] = deriv_raw[i, :, :]
            self.stored_derxyz = xyz.copy()
            self.stored_deriv = derivatives.copy()
            return derivatives
        
    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1, 3)
        if np.max(np.abs(xyz-self.stored_deriv2xyz)) < 1e-12:
            return self.stored_deriv2
        else:
            xsel = xyz[self.a, :]
            ysel = self.x0[self.a, :]
            xmean = np.mean(xsel,axis=0)
            ymean = np.mean(ysel,axis=0)
            deriv_raw, deriv2_raw = get_expmap_der(xsel, ysel, second=True, r=self.rnorm*self.rquat)
            second_derivatives = np.zeros((xyz.shape[0], 3, xyz.shape[0], 3, 3), dtype=float)
            for i, a in enumerate(self.a):
                for j, b in enumerate(self.a):
                    second_derivatives[a, :, b, :, :] = deriv2_raw[i, :, j, :, :]
            # derivatives, second_derivatives = get_expmap_der(xsel, ysel, second=True, r=self.rnorm*self.rquat)
            return second_derivatives
        
class RotationA(PrimitiveCoordinate):
    def __init__(self, a, x0, Rotators, w=1.0):
        self.a = tuple(sorted(a))
        self.x0 = x0
        self.w = w
        if self.a not in Rotators:
            Rotators[self.a] = Rotator(self.a, x0)
        self.Rotator = Rotators[self.a]
        self.isAngular = True
        self.isPeriodic = False

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

    def calcDiff(self, xyz1, xyz2=None, val2=None):
        return self.Rotator.calcDiff(xyz1, xyz2, val2)[0]*self.w
        
    def derivative(self, xyz):
        der_all = self.Rotator.derivative(xyz)
        derivatives = der_all[:, :, 0]*self.w
        return derivatives

    def second_derivative(self, xyz):
        deriv2_all = self.Rotator.second_derivative(xyz)
        second_derivatives = deriv2_all[:, :, :, :, 0]*self.w
        return second_derivatives
    
class RotationB(PrimitiveCoordinate):
    def __init__(self, a, x0, Rotators, w=1.0):
        self.a = tuple(sorted(a))
        self.x0 = x0
        self.w = w
        if self.a not in Rotators:
            Rotators[self.a] = Rotator(self.a, x0)
        self.Rotator = Rotators[self.a]
        self.isAngular = True
        self.isPeriodic = False

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
    
    def calcDiff(self, xyz1, xyz2=None, val2=None):
        return self.Rotator.calcDiff(xyz1, xyz2, val2)[1]*self.w
        
    def derivative(self, xyz):
        der_all = self.Rotator.derivative(xyz)
        derivatives = der_all[:, :, 1]*self.w
        return derivatives

    def second_derivative(self, xyz):
        deriv2_all = self.Rotator.second_derivative(xyz)
        second_derivatives = deriv2_all[:, :, :, :, 1]*self.w
        return second_derivatives

class RotationC(PrimitiveCoordinate):
    def __init__(self, a, x0, Rotators, w=1.0):
        self.a = tuple(sorted(a))
        self.x0 = x0
        self.w = w
        if self.a not in Rotators:
            Rotators[self.a] = Rotator(self.a, x0)
        self.Rotator = Rotators[self.a]
        self.isAngular = True
        self.isPeriodic = False

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
        
    def calcDiff(self, xyz1, xyz2=None, val2=None):
        return self.Rotator.calcDiff(xyz1, xyz2, val2)[2]*self.w

    def derivative(self, xyz):
        der_all = self.Rotator.derivative(xyz)
        derivatives = der_all[:, :, 2]*self.w
        return derivatives

    def second_derivative(self, xyz):
        deriv2_all = self.Rotator.second_derivative(xyz)
        second_derivatives = deriv2_all[:, :, :, :, 2]*self.w
        return second_derivatives

class Distance(PrimitiveCoordinate):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        if a == b:
            raise RuntimeError('a and b must be different')
        self.isAngular = False
        self.isPeriodic = False

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

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        m = self.a
        n = self.b
        l = np.linalg.norm(xyz[m] - xyz[n])
        u = (xyz[m] - xyz[n]) / l
        mtx = (np.outer(u, u) - np.eye(3))/l
        deriv2[m, :, m, :] = -mtx
        deriv2[n, :, n, :] = -mtx
        deriv2[m, :, n, :] = mtx
        deriv2[n, :, m, :] = mtx
        return deriv2
    
class Angle(PrimitiveCoordinate):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.isAngular = True
        self.isPeriodic = False
        if len({a, b, c}) != 3:
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
        # Catch the edge case that very rarely this number is -1.
        if dot / (norm1 * norm2) <= -1.0:
            if (np.abs(dot / (norm1 * norm2)) + 1.0) < -1e-6:
                raise RuntimeError('Encountered invalid value in angle')
            return np.pi
        if dot / (norm1 * norm2) >= 1.0:
            if (np.abs(dot / (norm1 * norm2)) - 1.0) > 1e-6:
                raise RuntimeError('Encountered invalid value in angle')
            return 0.0
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

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
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
        # Deriv2 derivatives are set to zero in the case of parallel or antiparallel vectors
        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            return deriv2
        # cosine and sine of the bond angle
        cq = np.dot(u, v)
        sq = np.sqrt(1-cq**2)
        uu = np.outer(u, u)
        uv = np.outer(u, v)
        vv = np.outer(v, v)
        de = np.eye(3)
        term1 = (uv + uv.T - (3*uu - de)*cq)/(u_norm**2*sq)
        term2 = (uv + uv.T - (3*vv - de)*cq)/(v_norm**2*sq)
        term3 = (uu + vv - uv*cq   - de)/(u_norm*v_norm*sq)
        term4 = (uu + vv - uv.T*cq - de)/(u_norm*v_norm*sq)
        der1 = self.derivative(xyz)
        def zeta(a_, m_, n_):
            return (int(a_==m_) - int(a_==n_))
        for a in [m, n, o]:
            for b in [m, n, o]:
                deriv2[a, :, b, :] = (zeta(a, m, o)*zeta(b, m, o)*term1
                                      + zeta(a, n, o)*zeta(b, n, o)*term2
                                      + zeta(a, m, o)*zeta(b, n, o)*term3
                                      + zeta(a, n, o)*zeta(b, m, o)*term4
                                      - (cq/sq) * np.outer(der1[a], der1[b]))
        return deriv2

class LinearAngle(PrimitiveCoordinate):
    def __init__(self, a, b, c, axis):
        self.a = a
        self.b = b
        self.c = c
        self.axis = axis
        self.isAngular = False
        self.isPeriodic = False
        if len({a, b, c}) != 3:
            raise RuntimeError('a, b, and c must be different')
        self.e0 = None
        self.stored_dot2 = 0.0

    def reset(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        # Cartesian axes.
        ex = np.array([1.0,0.0,0.0])
        ey = np.array([0.0,1.0,0.0])
        ez = np.array([0.0,0.0,1.0])
        self.e0 = [ex, ey, ez][np.argmin([np.dot(i, ev)**2 for i in [ex, ey, ez]])]
        self.stored_dot2 = 0.0

    def reposition_e0(self, xyz):
        """
        Project out the component of e0 that is parallel to ev. 
        This prevents linear angles from becoming parallel to e0, 
        which requires resetting the coordinate system.
        This function should be called at the end of each accepted step.
        """
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        dot = np.dot(ev, self.e0)
        self.e0 -= dot*ev
        self.e0 /= np.linalg.norm(self.e0)

    def __repr__(self):
        return "LinearAngle%s %i-%i-%i" % (["X","Y"][self.axis], self.a+1, self.b+1, self.c+1)

    def __eq__(self, other):
        if not hasattr(other, 'axis'): return False
        if self.axis is not other.axis: return False
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
        """
        This function measures the displacement of the BA and BC unit
        vectors in the linear angle "ABC". The displacements are measured
        along two axes that are perpendicular to the AC unit vector.
        """
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        if self.e0 is None: self.reset(xyz)
        e0 = self.e0
        self.stored_dot2 = np.dot(ev, e0)**2
        # Now make two unit vectors that are perpendicular to this one.
        c1 = np.cross(ev, e0)
        e1 = c1 / np.linalg.norm(c1)
        c2 = np.cross(ev, e1)
        e2 = c2 / np.linalg.norm(c2)
        # BA and BC unit vectors in ABC angle
        vba = xyz[a]-xyz[b]
        eba = vba / np.linalg.norm(vba)
        vbc = xyz[c]-xyz[b]
        ebc = vbc / np.linalg.norm(vbc)
        if self.axis == 0:
            answer = np.dot(eba, e1) + np.dot(ebc, e1)
        else:
            answer = np.dot(eba, e2) + np.dot(ebc, e2)
        return answer

    def visualize(self, xyz):
        xyz = xyz.reshape(-1,3)
        xsel = xyz[[self.a, self.b, self.c], :]
        xmean = np.mean(xsel,axis=0)
        a = self.a
        b = self.b
        c = self.c
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        if self.e0 is None: self.reset(xyz)
        e0 = self.e0
        self.stored_dot2 = np.dot(ev, e0)**2
        # Now make two unit vectors that are perpendicular to this one.
        c1 = np.cross(ev, e0)
        e1 = c1 / np.linalg.norm(c1)
        c2 = np.cross(ev, e1)
        e2 = c2 / np.linalg.norm(c2)
        # Visualize the rotated unit vectors.
        answer = np.zeros((3, 3), dtype=float)
        answer[0, :] = xmean
        answer[1, :] = xmean + e1*ang2bohr
        answer[2, :] = xmean + e2*ang2bohr
        return answer

    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        derivatives = np.zeros_like(xyz)
        ## Finite difference derivatives
        ## fderivatives = np.zeros_like(xyz)
        ## h = 1e-6
        ## for u in range(xyz.shape[0]):
        ##     for v in range(3):
        ##         xyz[u, v] += h
        ##         vPlus = self.value(xyz)
        ##         xyz[u, v] -= 2*h
        ##         vMinus = self.value(xyz)
        ##         xyz[u, v] += h
        ##         fderivatives[u, v] = (vPlus-vMinus)/(2*h)
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        if self.e0 is None: self.reset(xyz)
        e0 = self.e0
        c1 = np.cross(ev, e0)
        e1 = c1 / np.linalg.norm(c1)
        c2 = np.cross(ev, e1)
        e2 = c2 / np.linalg.norm(c2)
        # BA and BC unit vectors in ABC angle
        vba = xyz[a]-xyz[b]
        eba = vba / np.linalg.norm(vba)
        vbc = xyz[c]-xyz[b]
        ebc = vbc / np.linalg.norm(vbc)
        # Derivative terms
        de0 = np.zeros((3, 3), dtype=float)
        dev = d_unit_vector(v)
        dc1 = d_cross_ab(ev, e0, dev, de0)
        de1 = np.dot(dc1, d_unit_vector(c1))
        dc2 = d_cross_ab(ev, e1, dev, de1)
        de2 = np.dot(dc2, d_unit_vector(c2))
        deba = d_unit_vector(vba)
        debc = d_unit_vector(vbc)
        if self.axis == 0:
            derivatives[a, :] = np.dot(deba, e1) + np.dot(-de1, eba) + np.dot(-de1, ebc)
            derivatives[b, :] = np.dot(-deba, e1) + np.dot(-debc, e1)
            derivatives[c, :] = np.dot(de1, eba) + np.dot(de1, ebc) + np.dot(debc, e1)
        else:
            derivatives[a, :] = np.dot(deba, e2) + np.dot(-de2, eba) + np.dot(-de2, ebc)
            derivatives[b, :] = np.dot(-deba, e2) + np.dot(-debc, e2)
            derivatives[c, :] = np.dot(de2, eba) + np.dot(de2, ebc) + np.dot(debc, e2)
        ## Finite difference derivatives
        ## if np.linalg.norm(derivatives - fderivatives) > 1e-6:
        ##     print np.linalg.norm(derivatives - fderivatives)
        ##     raise Exception()
        return derivatives
    
    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        deriv2 = np.zeros((xyz.shape[0], 3, xyz.shape[0], 3), dtype=float)
        h = 1.0e-3
        for i in range(3):
            for j in range(3):
                ii = [a, b, c][i]
                xyz[ii, j] += h
                FPlus = self.derivative(xyz)
                xyz[ii, j] -= 2*h
                FMinus = self.derivative(xyz)
                xyz[ii, j] += h
                fderiv = (FPlus-FMinus)/(2*h)
                deriv2[ii, j, :, :] = fderiv
        return deriv2
    
class MultiAngle(PrimitiveCoordinate): # pragma: no cover
    def __init__(self, a, b, c):
        if type(a) is int:
            a = (a,)
        if type(c) is int:
            c = (c,)
        self.a = tuple(a)
        self.b = b
        self.c = tuple(c)
        self.isAngular = True
        self.isPeriodic = False
        if len({a, b, c}) != 3:
            raise RuntimeError('a, b, and c must be different')

    def __repr__(self):
        stra = ("("+','.join(["%i" % (i+1) for i in self.a])+")") if len(self.a) > 1 else "%i" % (self.a[0]+1)
        strc = ("("+','.join(["%i" % (i+1) for i in self.c])+")") if len(self.c) > 1 else "%i" % (self.c[0]+1)
        return "%sAngle %s-%i-%s" % ("Multi" if (len(self.a) > 1 or len(self.c) > 1) else "", stra, self.b+1, strc)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.b == other.b:
            if set(self.a) == set(other.a):
                if set(self.c) == set(other.c):
                    return True
            if set(self.a) == set(other.c):
                if set(self.c) == set(other.a):
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        b = self.b
        c = np.array(self.c)
        xyza = np.mean(xyz[a], axis=0)
        xyzc = np.mean(xyz[c], axis=0)
        # vector from first atom to central atom
        vector1 = xyza - xyz[b]
        # vector from last atom to central atom
        vector2 = xyzc - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        dot = np.dot(vector1, vector2)
        # Catch the edge case that very rarely this number is -1.
        if dot / (norm1 * norm2) <= -1.0:
            if (np.abs(dot / (norm1 * norm2)) + 1.0) < -1e-6:
                raise RuntimeError('Encountered invalid value in angle')
            return np.pi
        return np.arccos(dot / (norm1 * norm2))

    def normal_vector(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        b = self.b
        c = np.array(self.c)
        xyza = np.mean(xyz[a], axis=0)
        xyzc = np.mean(xyz[c], axis=0)
        # vector from first atom to central atom
        vector1 = xyza - xyz[b]
        # vector from last atom to central atom
        vector2 = xyzc - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        crs = np.cross(vector1, vector2)
        crs /= np.linalg.norm(crs)
        return crs
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = np.array(self.a)
        o = self.b
        n = np.array(self.c)
        xyzm = np.mean(xyz[m], axis=0)
        xyzn = np.mean(xyz[n], axis=0)
        # Unit displacement vectors
        u_prime = (xyzm - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyzn - xyz[o])
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
        for i in m:
            derivatives[i, :] = term1/len(m)
        for i in n:
            derivatives[i, :] = term2/len(n)
        derivatives[o, :] = -(term1 + term2)
        return derivatives
    
    def second_derivative(self, xyz):
        raise NotImplementedError("Second derivatives have not been implemented for IC type %s" % self.__name__)

class Dihedral(PrimitiveCoordinate):
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.isAngular = True
        self.isPeriodic = True
        if len({a, b, c, d}) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        return "Dihedral %i-%i-%i-%i" % (self.a+1, self.b+1, self.c+1, self.d+1)

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

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        m = self.a
        o = self.b
        p = self.c
        n = self.d
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        lu = np.linalg.norm(u_prime)
        lw = np.linalg.norm(w_prime)
        lv = np.linalg.norm(v_prime)
        u = u_prime / lu
        w = w_prime / lw
        v = v_prime / lv
        cu = np.dot(u, w)
        su = (1 - np.dot(u, w)**2)**0.5
        su4 = su**4
        cv = np.dot(v, w)
        sv = (1 - np.dot(v, w)**2)**0.5
        sv4 = sv**4
        if su < 1e-6 or sv < 1e-6 : return deriv2
        
        uxw = np.cross(u, w)
        vxw = np.cross(v, w)

        term1 = np.outer(uxw, w*cu - u)/(lu**2*su4)
        term2 = np.outer(vxw, -w*cv + v)/(lv**2*sv4)
        term3 = np.outer(uxw, w - 2*u*cu + w*cu**2)/(2*lu*lw*su4)
        term4 = np.outer(vxw, w - 2*v*cv + w*cv**2)/(2*lv*lw*sv4)
        term5 = np.outer(uxw, u + u*cu**2 - 3*w*cu + w*cu**3)/(2*lw**2*su4)
        term6 = np.outer(vxw,-v - v*cv**2 + 3*w*cv - w*cv**3)/(2*lw**2*sv4)
        term1 += term1.T
        term2 += term2.T
        term3 += term3.T
        term4 += term4.T
        term5 += term5.T
        term6 += term6.T
        def mk_amat(vec):
            amat = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    if i == j: continue
                    k = 3 - i - j
                    amat[i, j] = vec[k] * (j-i) * ((-0.5)**np.abs(j-i))
            return amat
        term7 = mk_amat((-w*cu + u)/(lu*lw*su**2))
        term8 = mk_amat(( w*cv - v)/(lv*lw*sv**2))
        def zeta(a_, m_, n_):
            return (int(a_==m_) - int(a_==n_))
        # deriv2_terms = [np.zeros_like(deriv2) for i in range(9)]
        # Accumulate the second derivative
        for a in [m, n, o, p]:
            for b in [m, n, o, p]:
                deriv2[a, :, b, :] = (zeta(a, m, o)*zeta(b, m, o)*term1 +
                                      zeta(a, n, p)*zeta(b, n, p)*term2 +
                                      (zeta(a, m, o)*zeta(b, o, p) + zeta(a, p, o)*zeta(b, o, m))*term3 +
                                      (zeta(a, n, p)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, n, p))*term4 +
                                      zeta(a, o, p)*zeta(b, p, o)*term5 +
                                      zeta(a, p, o)*zeta(b, o, p)*term6)
                if a != b:
                    deriv2[a, :, b, :] += ((zeta(a, m, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, m))*term7 +
                                           (zeta(a, n, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, n))*term8)
        return deriv2
                    
        # Accumulate a dictionary of contributions to the second derivatives by term (for debugging)
        #             deriv2_terms[7][a, :, b, :] = (zeta(a, m, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, m))*term7
        #             deriv2_terms[8][a, :, b, :] = (zeta(a, n, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, n))*term8
        #         deriv2_terms[1][a, :, b, :] = zeta(a, m, o)*zeta(b, m, o)*term1
        #         deriv2_terms[2][a, :, b, :] = zeta(a, n, p)*zeta(b, n, p)*term2
        #         deriv2_terms[3][a, :, b, :] = (zeta(a, m, o)*zeta(b, o, p) + zeta(a, p, o)*zeta(b, o, m))*term3
        #         deriv2_terms[4][a, :, b, :] = (zeta(a, n, p)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, n, p))*term4
        #         deriv2_terms[5][a, :, b, :] = zeta(a, o, p)*zeta(b, p, o)*term5
        #         deriv2_terms[6][a, :, b, :] = zeta(a, p, o)*zeta(b, o, p)*term6
        # deriv2_terms[0] = deriv2.copy()
        # 
        #=======
        # Term-by-term checking of the second derivative.
        # Produces output such as:
        # 1x1x a:  0.0000 n:  0.0000 e:  0.0000 Terms: NNNNNNNN  0.0000  0.0000 -0.0000 -0.0000 -0.0000 -0.0000  0.0000  0.0000
        # 1x1y a:  0.3337 n:  0.3337 e:  0.0000 Terms: YNNNNNNN  0.3337  0.0000 -0.0000 -0.0000 -0.0000 -0.0000  0.0000  0.0000
        # 1x1z a:  0.0590 n:  0.0590 e: -0.0000 Terms: YNNNNNNN  0.0590  0.0000 -0.0000  0.0000 -0.0000  0.0000  0.0000  0.0000
        # 
        # def printTerm(strin, num):
        #     i = int(strin[0])-1
        #     j = 'xyz'.index(strin[1])
        #     k = int(strin[2])-1
        #     l = 'xyz'.index(strin[3])
        #     ana = deriv2_terms[0][i,j,k,l]
        #     err = ana-num
        #     correct = np.abs(num-ana) < 1e-5
        #     color = '\x1b[92m' if correct else '\x1b[91m'
        #     print('%i%s%i%s a: % .4f n: % .4f e: % .4f Terms: ' % (i+1, 'xyz'[j], k+1, 'xyz'[l], ana, num, err) +
        #           ''.join(["Y" if np.abs(deriv2_terms[m][i,j,k,l]) > 1e-5 else "N" for m in range(1, 9)]) + ' ' +
        #           ' '.join(["%s% .4f\x1b[0m" % (color if np.abs(deriv2_terms[m][i,j,k,l]) > 1e-5 else '',
        #                                         deriv2_terms[m][i,j,k,l]) for m in range(1, 9)]))
        # print("LP checking single term:")
        # printTerm('1x1x',  5.55112e-09)
        # printTerm('1x1y',  3.33702e-01)
        # printTerm('1x1z',  5.90389e-02)

class MultiDihedral(PrimitiveCoordinate): # pragma: no cover
    def __init__(self, a, b, c, d):
        if type(a) is int:
            a = (a, )
        if type(d) is int:
            d = (d, )
        self.a = tuple(a)
        self.b = b
        self.c = c
        self.d = tuple(d)
        self.isAngular = True
        self.isPeriodic = True
        if len({a, b, c, d}) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        stra = ("("+','.join(["%i" % (i+1) for i in self.a])+")") if len(self.a) > 1 else "%i" % (self.a[0]+1)
        strd = ("("+','.join(["%i" % (i+1) for i in self.d])+")") if len(self.d) > 1 else "%i" % (self.d[0]+1)
        return "%sDihedral %s-%i-%i-%s" % ("Multi" if (len(self.a) > 1 or len(self.d) > 1) else "", stra, self.b+1, self.c+1, strd)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if set(self.a) == set(other.a):
            if self.b == other.b:
                if self.c == other.c:
                    if set(self.d) == set(other.d):
                        return True
        if set(self.a) == set(other.d):
            if self.b == other.c:
                if self.c == other.b:
                    if set(self.d) == set(other.a):
                        return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        b = self.b
        c = self.c
        d = np.array(self.d)
        xyza = np.mean(xyz[a], axis=0)
        xyzd = np.mean(xyz[d], axis=0)
        
        vec1 = xyz[b] - xyza
        vec2 = xyz[c] - xyz[b]
        vec3 = xyzd - xyz[c]
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
        m = np.array(self.a)
        o = self.b
        p = self.c
        n = np.array(self.d)
        xyzm = np.mean(xyz[m], axis=0)
        xyzn = np.mean(xyz[n], axis=0)
        
        u_prime = (xyzm - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyzn - xyz[p])
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
        for i in self.a:
            derivatives[i, :] = term1/len(self.a)
        for i in self.d:
            derivatives[i, :] = -term2/len(self.d)
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives
    
    def second_derivative(self, xyz):
        raise NotImplementedError("Second derivatives have not been implemented for IC type %s" % self.__name__)

class OutOfPlane(PrimitiveCoordinate):
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.isAngular = True
        self.isPeriodic = True
        if len({a, b, c, d}) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        return "Out-of-Plane %i-%i-%i-%i" % (self.a+1, self.b+1, self.c+1, self.d+1)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if {self.b, self.c, self.d} == {other.b, other.c, other.d}:
                if [self.b, self.c, self.d] != [other.b, other.c, other.d]:
                    logger.warning("Warning: OutOfPlane atoms are the same, ordering is different\n")
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

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        deriv2 = np.zeros((xyz.shape[0], 3, xyz.shape[0], 3), dtype=float)
        h = 1.0e-3
        for i in range(4):
            for j in range(3):
                ii = [a, b, c, d][i]
                xyz[ii, j] += h
                FPlus = self.derivative(xyz)
                xyz[ii, j] -= 2*h
                FMinus = self.derivative(xyz)
                xyz[ii, j] += h
                fderiv = (FPlus-FMinus)/(2*h)
                deriv2[ii, j, :, :] = fderiv
        return deriv2

def convert_angstroms_degrees(prims, values):
    """ Convert values of primitive ICs (or differences) from
    weighted atomic units to Angstroms and degrees. """
    converted = np.array(values).copy()
    for ic, c in enumerate(prims):
        if type(c) in [TranslationX, TranslationY, TranslationZ]:
            w = 1.0
        elif hasattr(c, 'w'):
            w = c.w
        else:
            w = 1.0
        if type(c) in [TranslationX, TranslationY, TranslationZ, CartesianX, CartesianY, CartesianZ, Distance, LinearAngle]:
            factor = bohr2ang
        elif c.isAngular:
            factor = 180.0 / np.pi
        converted[ic] /= w
        converted[ic] *= factor
    return converted

CacheWarning = False

class InternalCoordinates(object):
    def __init__(self):
        self.stored_wilsonB = OrderedDict()

    def addConstraint(self, cPrim, cVal):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def haveConstraints(self):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def augmentGH(self, xyz, G, H):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def calcGradProj(self, xyz, gradx):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def clearCache(self):
        self.stored_wilsonB = OrderedDict()

    def wilsonB(self, xyz):
        """
        Given Cartesian coordinates xyz, return the Wilson B-matrix
        given by dq_i/dx_j where x is flattened (i.e. x1, y1, z1, x2, y2, z2)
        """
        global CacheWarning
        t0 = time.time()
        xhash = hash(xyz.tostring())
        ht = time.time() - t0
        if xhash in self.stored_wilsonB:
            ans = self.stored_wilsonB[xhash]
            return ans
        WilsonB = []
        Der = self.derivatives(xyz)
        for i in range(Der.shape[0]):
            WilsonB.append(Der[i].flatten())
        self.stored_wilsonB[xhash] = np.array(WilsonB)
        if len(self.stored_wilsonB) > 1000 and not CacheWarning:
            logger.warning("\x1b[91mWarning: more than 1000 B-matrices stored, memory leaks likely\x1b[0m\n")
            CacheWarning = True
        ans = np.array(WilsonB)
        return ans

    def GMatrix(self, xyz):
        """
        Given Cartesian coordinates xyz, return the G-matrix
        given by G = BuBt where u is an arbitrary matrix (default to identity)
        """
        Bmat = self.wilsonB(xyz)
        BuBt = np.dot(Bmat,Bmat.T)
        return BuBt

    def GInverse_SVD(self, xyz):
        xyz = xyz.reshape(-1,3)
        # Perform singular value decomposition
        click()
        loops = 0
        while True:
            try:
                G = self.GMatrix(xyz)
                time_G = click()
                U, S, VT = np.linalg.svd(G)
                time_svd = click()
            except np.linalg.LinAlgError:
                logger.warning("\x1b[1;91m SVD fails, perturbing coordinates and trying again\x1b[0m\n")
                xyz = xyz + 1e-2*np.random.random(xyz.shape)
                loops += 1
                if loops == 10:
                    raise RuntimeError('SVD failed too many times')
                continue
            break
        # print "Build G: %.3f SVD: %.3f" % (time_G, time_svd),
        V = VT.T
        UT = U.T
        Sinv = np.zeros_like(S)
        LargeVals = 0
        for ival, value in enumerate(S):
            # print "%.5e % .5e" % (ival,value)
            if np.abs(value) > 1e-6:
                LargeVals += 1
                Sinv[ival] = 1/value
        # print "%i atoms; %i/%i singular values are > 1e-6" % (xyz.shape[0], LargeVals, len(S))
        Sinv = np.diag(Sinv)
        Inv = multi_dot([V, Sinv, UT])
        return Inv

    def GInverse_EIG(self, xyz):
        xyz = xyz.reshape(-1,3)
        click()
        G = self.GMatrix(xyz)
        time_G = click()
        Gi = np.linalg.inv(G)
        time_inv = click()
        # print "G-time: %.3f Inv-time: %.3f" % (time_G, time_inv)
        return Gi

    def checkFiniteDifferenceGrad(self, xyz):
        xyz = xyz.reshape(-1,3)
        Analytical = self.derivatives(xyz)
        FiniteDifference = np.zeros_like(Analytical)
        h = 1e-5
        for i in range(xyz.shape[0]):
            for j in range(3):
                x1 = xyz.copy()
                x2 = xyz.copy()
                x1[i,j] += h
                x2[i,j] -= h
                PMDiff = self.calcDiff(x1,x2)
                FiniteDifference[:,i,j] = PMDiff/(2*h)
        logger.info("-=# Now checking first derivatives of internal coordinates w/r.t. Cartesians #=-\n")
        for i in range(Analytical.shape[0]):
            title = "%20s : %20s" % ("IC %i/%i" % (i+1, Analytical.shape[0]), self.Internals[i])
            lines = [title]
            maxerr = 0.0
            for j in range(Analytical.shape[1]):
                lines.append("Atom %i" % (j+1))
                for k in range(Analytical.shape[2]):
                    error = Analytical[i,j,k] - FiniteDifference[i,j,k]
                    if np.abs(error) > 1e-5:
                        color = "\x1b[91m"
                    else:
                        color = "\x1b[92m"
                    lines.append("%s % .5e % .5e %s% .5e\x1b[0m" % ("xyz"[k], Analytical[i,j,k], FiniteDifference[i,j,k], color, Analytical[i,j,k] - FiniteDifference[i,j,k]))
                    if maxerr < np.abs(error):
                        maxerr = np.abs(error)
            if maxerr > 1e-5:
                logger.info('\n'.join(lines)+'\n')
            logger.info("%s : Max Error = %.5e\n" % (title, maxerr))
        logger.info("Finite-difference Finished\n")
        return FiniteDifference

    def checkFiniteDifferenceHess(self, xyz):
        xyz = xyz.reshape(-1,3)
        Analytical = self.second_derivatives(xyz)
        FiniteDifference = np.zeros_like(Analytical)
        h = 1e-4
        verbose = False
        logger.info("-=# Now checking second derivatives of internal coordinates w/r.t. Cartesians #=-\n")
        for j in range(xyz.shape[0]):
            for m in range(3):
                for k in range(xyz.shape[0]):
                    for n in range(3):
                        x1 = xyz.copy()
                        x2 = xyz.copy()
                        x3 = xyz.copy()
                        x4 = xyz.copy()
                        x1[j, m] += h
                        x1[k, n] += h # (+, +)
                        x2[j, m] += h
                        x2[k, n] -= h # (+, -)
                        x3[j, m] -= h
                        x3[k, n] += h # (-, +)
                        x4[j, m] -= h
                        x4[k, n] -= h # (-, -)
                        PMDiff1 = self.calcDiff(x1, x2)
                        PMDiff2 = self.calcDiff(x4, x3)
                        FiniteDifference[:, j, m, k, n] += (PMDiff1+PMDiff2)/(4*h**2)
        #                 print('\r%i %i' % (j, k), end='')
        # print()
        for i in range(Analytical.shape[0]):
            title = "%20s : %20s" % ("IC %i/%i" % (i+1, Analytical.shape[0]), self.Internals[i])
            lines = [title]
            if verbose: logger.info(title+'\n')
            maxerr = 0.0
            numerr = 0
            for j in range(Analytical.shape[1]):
                for m in range(Analytical.shape[2]):
                    for k in range(Analytical.shape[3]):
                        for n in range(Analytical.shape[4]):
                            ana = Analytical[i,j,m,k,n]
                            fin = FiniteDifference[i,j,m,k,n]
                            error = ana - fin
                            message = "Atom %i %s %i %s a: % 12.5e n: % 12.5e e: % 12.5e %s" % (j+1, 'xyz'[m], k+1, 'xyz'[n], ana, fin,
                                                                                                error, 'X' if np.abs(error)>1e-5 else '')
                            if np.abs(error)>1e-5:
                                numerr += 1
                            if (ana != 0.0 or fin != 0.0) and verbose:
                                logger.info(message+'\n')
                            lines.append(message)
                            if maxerr < np.abs(error):
                                maxerr = np.abs(error)
            if maxerr > 1e-5 and not verbose:
                logger.info('\n'.join(lines)+'\n')
            logger.info("%s : Max Error = % 12.5e (%i above threshold)\n" % (title, maxerr, numerr))
        logger.info("Finite-difference Finished\n")
        return FiniteDifference
        
    def calcGrad(self, xyz, gradx):
        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
        Gq = multi_dot([Ginv, Bmat, gradx.T])
        return Gq.flatten()

    def calcHess(self, xyz, gradx, hessx):
        """
        Compute the internal coordinate Hessian. 
        Expects Cartesian coordinates to be provided in a.u.
        """
        xyz = xyz.flatten()
        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        Gq = self.calcGrad(xyz, gradx)
        deriv2 = self.second_derivatives(xyz)
        Bmatp = deriv2.reshape(deriv2.shape[0], xyz.shape[0], xyz.shape[0])
        Hx_BptGq = hessx - np.einsum('pmn,p->mn',Bmatp,Gq)
        Hq = np.einsum('ps,sm,mn,nr,rq', Ginv, Bmat, Hx_BptGq, Bmat.T, Ginv, optimize=True)
        return Hq

    def calcHessCart(self, xyz, gradq, hessq):
        """
        Compute the Cartesian Hessian given internal coordinate gradient and Hessian. 
        Returns the answer in a.u.
        """
        xyz = xyz.flatten()
        Bmat = self.wilsonB(xyz)
        deriv2 = self.second_derivatives(xyz)
        Bmatp = deriv2.reshape(deriv2.shape[0], xyz.shape[0], xyz.shape[0])
        BptGq = np.einsum('pmn,p->mn',Bmatp,gradq)
        Hx = np.einsum('ai,ab,bj->ij', Bmat, hessq, Bmat, optimize=True)
        Hx += BptGq
        return Hx
    
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

    def newCartesian(self, xyz, dQ, verbose=True):
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
        if verbose >= 2: logger.info("    InternalCoordinates.newCartesian converting internal to Cartesian step\n")
        def finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1):
            if ndqt > 1e-1:
                if verbose: logger.info("      newCartesian Iter: %i Failed to obtain coordinates (rmsd = %.3e |dQ| = %.3e)\n" % (microiter, rmsdt, ndqt))
                self.bork = True
                self.writeCache(xyz, dQ, xyz_iter1)
                return xyz_iter1.flatten()
            elif ndqt > 1e-3:
                if verbose: logger.info("      newCartesian Iter: %i Approximate coordinates obtained (rmsd = %.3e |dQ| = %.3e)\n" % (microiter, rmsdt, ndqt))
            else:
                if verbose: logger.info("      newCartesian Iter: %i Cartesian coordinates obtained (rmsd = %.3e |dQ| = %.3e)\n" % (microiter, rmsdt, ndqt))
            self.writeCache(xyz, dQ, xyzsave)
            return xyzsave.flatten()
        fail_counter = 0
        while True:
            microiter += 1
            Bmat = self.wilsonB(xyz1)
            Ginv = self.GInverse(xyz1)
            # Get new Cartesian coordinates
            dxyz = damp*multi_dot([Bmat.T,Ginv,dQ1.T])
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
                    if verbose >= 2: logger.info("      newCartesian Iter: %i Err-dQ (Best) = %.5e (%.5e) RMSD: %.5e Damp: %.5e (Bad)\n" % (microiter, ndq, ndqt, rmsd, damp))
                    damp /= 2
                    fail_counter += 1
                    # xyz2 = xyz1.copy()
                else:
                    if verbose >= 2: logger.info("      newCartesian Iter: %i Err-dQ (Best) = %.5e (%.5e) RMSD: %.5e Damp: %.5e (Good)\n" % (microiter, ndq, ndqt, rmsd, damp))
                    fail_counter = 0
                    damp = min(damp*1.2, 1.0)
                    rmsdt = rmsd
                    ndqt = ndq
                    xyzsave = xyz2.copy()
            else:
                if verbose >= 2: logger.info("      newCartesian Iter: %i Err-dQ = %.5e RMSD: %.5e Damp: %.5e\n" % (microiter, ndq, rmsd, damp))
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
    def __init__(self, molecule, connect=False, addcart=False, constraints=None, cvals=None, **kwargs):
        super(PrimitiveInternalCoordinates, self).__init__()
        self.connect = connect
        self.addcart = addcart
        self.Internals = []
        self.cPrims = []
        self.cVals = []
        self.Rotators = OrderedDict()
        self.elem = molecule.elem
        # List of fragments as determined by residue ID, distance criteria or bond order
        self.frags = []
        for i in range(len(molecule)):
            self.makePrimitives(molecule[i], connect, addcart)
        # Assume we're using the first image for constraints
        self.makeConstraints(molecule[0], constraints, cvals)
        # Reorder primitives for checking with cc's code in TC.
        # Note that reorderPrimitives() _must_ be updated with each new InternalCoordinate class written.
        self.reorderPrimitives()

    def makePrimitives(self, molecule, connect, addcart):
        # force_bonds=False is set because we don't want to override
        # bond order-based bonds that may have been obtained earlier.
        molecule.build_topology(force_bonds=False)
        connect_isolated = True
        if 'resid' in molecule.Data.keys():
            # Create fragments corresponding to unique resID numbers if provided.
            frags = []
            residues = []
            current_resid = -1
            for i in range(molecule.na):
                if molecule.resid[i] != current_resid:
                    residues.append([i])
                    current_resid = molecule.resid[i]
                else:
                    residues[-1].append(i)
            # A single residue is not always guaranteed to be contiguous
            for res in residues:
                residue_select = molecule.atom_select(res)
                residue_select.build_topology(force_bonds=False)
                for sub_mol in residue_select.molecules:
                    frags.append([res[i] for i in sub_mol])
        else:
            # Create fragments based on connectivity in provided molecule object.
            frags = [list(m.nodes()) for m in molecule.molecules]
            if connect_isolated:
                isolates = [list(m.nodes())[0] for m in molecule.molecules if len(m.nodes()) == 1]
                for i in isolates:
                    j = molecule.get_closest_atom(i, pbc=False)[0]
                    logger.info("Creating artifical bond to isolated atom: %i-%i\n" % (i+1, j+1))
                    molecule.bonds.append((i, j))
                old_read_bonds = molecule.top_settings['read_bonds']
                molecule.top_settings['read_bonds'] = True
                molecule.build_topology(force_bonds=False)
                molecule.top_settings['read_bonds'] = old_read_bonds
                frags = [list(m.nodes()) for m in molecule.molecules]
        # Make frags accessible from outside.
        self.frags = frags
            
        # coordinates in Angstrom
        coords = molecule.xyzs[0].flatten()
        # Make a distance matrix mapping atom pairs to interatomic distances
        AtomIterator, dxij = molecule.distance_matrix(pbc=False)
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
            # LPW 2022-08-12
            # Technically the MST should not add edges within a molecule, but
            # the "bug" actually seemed to improve performance for the constrained
            # dipeptide case (from OpenFF). To "fix" the bug add the following 
            # clause to the if statement on the next line:
            # and not nx.has_path(molecule.topology, edge[0], edge[1]):
            if edge not in list(molecule.topology.edges()):
                # print "Adding %s from minimum spanning tree" % str(edge)
                if connect:
                    molecule.topology.add_edge(edge[0], edge[1])
                    noncov.append(edge)
        if not connect:
            if addcart:
                for i in range(molecule.na):
                    self.add(CartesianX(i, w=1.0))
                    self.add(CartesianY(i, w=1.0))
                    self.add(CartesianZ(i, w=1.0))
            else:
                for i in frags:
                    if len(i) >= 2:
                        self.add(TranslationX(i, w=np.ones(len(i))/len(i)))
                        self.add(TranslationY(i, w=np.ones(len(i))/len(i)))
                        self.add(TranslationZ(i, w=np.ones(len(i))/len(i)))
                        # Reference coordinates are given in Bohr.
                        sel = coords.reshape(-1,3)[i,:] * ang2bohr
                        sel -= np.mean(sel, axis=0)
                        rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
                        self.add(RotationA(i, coords * ang2bohr, self.Rotators, w=rg))
                        self.add(RotationB(i, coords * ang2bohr, self.Rotators, w=rg))
                        self.add(RotationC(i, coords * ang2bohr, self.Rotators, w=rg))
                    else:
                        for j in i:
                            self.add(CartesianX(j, w=1.0))
                            self.add(CartesianY(j, w=1.0))
                            self.add(CartesianZ(j, w=1.0))
        add_tr = False
        if add_tr:
            i = range(molecule.na)
            self.add(TranslationX(i, w=np.ones(len(i))/len(i)))
            self.add(TranslationY(i, w=np.ones(len(i))/len(i)))
            self.add(TranslationZ(i, w=np.ones(len(i))/len(i)))
            # Reference coordinates are given in Bohr.
            sel = coords.reshape(-1,3)[i,:] * ang2bohr
            sel -= np.mean(sel, axis=0)
            rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
            self.add(RotationA(i, coords * ang2bohr, self.Rotators, w=rg))
            self.add(RotationB(i, coords * ang2bohr, self.Rotators, w=rg))
            self.add(RotationC(i, coords * ang2bohr, self.Rotators, w=rg))

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

        # Add an internal coordinate for bonded atom pairs
        for (a, b) in molecule.topology.edges():
            self.add(Distance(a, b))

        # Linear angle threshold - corresponds to about 162 degrees.
        LinThre   = 0.95
        AngDict = defaultdict(list)
        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    if a < c:
                        # if (a, c) in molecule.topology.edges() or (c, a) in molecule.topology.edges(): continue
                        Ang = Angle(a, b, c)
                        nnc = (min(a, b), max(a, b)) in noncov
                        nnc += (min(b, c), max(b, c)) in noncov
                        # if nnc >= 2: continue
                        # logger.info(" >>> Linear angle check: %3i %3i %3i % .3f % .3f %s\n" % (a, b, c, Ang.value(coords)*180/np.pi, np.cos(Ang.value(coords)), str(np.abs(np.cos(Ang.value(coords))) > LinThre)))
                        if np.abs(np.cos(Ang.value(coords))) < LinThre:
                            self.add(Angle(a, b, c))
                            AngDict[b].append(Ang)
                        elif connect or not addcart:
                            # logger.info("Adding linear angle")
                            # Add linear angle IC's
                            # LPW 2019-02-16: Linear angle ICs work well for "very" linear angles in molecules (e.g. HCCCN)
                            # but do not work well for "almost" linear angles in noncovalent systems (e.g. H2O6).
                            # Bringing back old code to use "translations" for the latter case, but should be investigated
                            # more deeply in the future.
                            # LPW 2022-02-15: Linear angle ICs have been improved, and should no longer require resetting if the
                            # atoms in the angle go through a large rotation. They are currently being used.
                            if nnc == 0:
                                self.add(LinearAngle(a, b, c, 0))
                                self.add(LinearAngle(a, b, c, 1))
                            else:
                                # Unit vector connecting atoms a and c
                                nac = molecule.xyzs[0][c] - molecule.xyzs[0][a]
                                nac /= np.linalg.norm(nac)
                                # Dot products of this vector with the Cartesian axes
                                dots = [np.abs(np.dot(ei, nac)) for ei in np.eye(3)]
                                # Functions for adding Cartesian coordinate
                                # carts = [CartesianX, CartesianY, CartesianZ]
                                trans = [TranslationX, TranslationY, TranslationZ]
                                w = np.array([-1.0, 2.0, -1.0])
                                # Add two of the most perpendicular Cartesian coordinates
                                for i in np.argsort(dots)[:2]:
                                    self.add(trans[i]([a, b, c], w=w))
                            
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
                                if np.abs(np.dot(Ang1.normal_vector(coords), Ang2.normal_vector(coords))) > LinThre:
                                    self.delete(Angle(i, b, j))
                                    self.add(OutOfPlane(b, i, j, k))
                                    break
                                
        # Find groups of atoms that are in straight lines
        atom_lines = [list(i) for i in molecule.topology.edges()]
        while True:
            # For a line of two atoms (one bond):
            # AB-AC
            # AX-AY
            # i.e. AB is the first one, AC is the second one
            # AX is the second-to-last one, AY is the last one
            # AB-AC-...-AX-AY
            # AB-(AC, AX)-AY
            atom_lines0 = deepcopy(atom_lines)
            for aline in atom_lines:
                # Imagine a line of atoms going like ab-ac-ax-ay.
                # Our job is to extend the line until there are no more
                ab = aline[0]
                ay = aline[-1]
                for aa in molecule.topology.neighbors(ab):
                    if aa not in aline:
                        # If the angle that AA makes with AB and ALL other atoms AC in the line are linear:
                        # Add AA to the front of the list
                        if all([np.abs(np.cos(Angle(aa, ab, ac).value(coords))) > LinThre for ac in aline[1:] if ac != ab]):
                            aline.insert(0, aa)
                for az in molecule.topology.neighbors(ay):
                    if az not in aline:
                        if all([np.abs(np.cos(Angle(ax, ay, az).value(coords))) > LinThre for ax in aline[:-1] if ax != ay]):
                            aline.append(az)
            if atom_lines == atom_lines0: break
        atom_lines_uniq = []
        for i in atom_lines:    # 
            if tuple(i) not in set(atom_lines_uniq):
                atom_lines_uniq.append(tuple(i))
        lthree = [l for l in atom_lines_uniq if len(l) > 2]
        # TODO: Perhaps should reduce the times this is printed out in reaction paths
        # if len(lthree) > 0:
        #     print "Lines of three or more atoms:", ', '.join(['-'.join(["%i" % (i+1) for i in l]) for l in lthree])

        # Normal dihedral code
        for aline in atom_lines_uniq:
            # Go over ALL pairs of atoms in a line
            for (b, c) in itertools.combinations(aline, 2):
                if b > c: (b, c) = (c, b)
                # Go over all neighbors of b
                for a in molecule.topology.neighbors(b):
                    # Go over all neighbors of c
                    for d in molecule.topology.neighbors(c):
                        # Make sure the end-atoms are not in the line and not the same as each other
                        if a not in aline and d not in aline and a != d:
                            nnc = (min(a, b), max(a, b)) in noncov
                            nnc += (min(b, c), max(b, c)) in noncov
                            nnc += (min(c, d), max(c, d)) in noncov
                            # print aline, a, b, c, d
                            Ang1 = Angle(a,b,c)
                            Ang2 = Angle(b,c,d)
                            # Eliminate dihedrals containing angles that are almost linear
                            # (should be eliminated already)
                            if np.abs(np.cos(Ang1.value(coords))) > LinThre: continue
                            if np.abs(np.cos(Ang2.value(coords))) > LinThre: continue
                            self.add(Dihedral(a, b, c, d))
            
        ### Following are codes that evaluate angles and dihedrals involving entire lines-of-atoms
        ### as single degrees of freedom
        ### Unfortunately, they do not seem to improve the performance
        #
        # def pull_lines(a, front=True, middle=False):
        #     """
        #     Given an atom, pull all lines-of-atoms that it is in, e.g.
        #     e.g. 
        #               D
        #               C
        #               B
        #           EFGHAIJKL
        #     returns (B, C, D), (H, G, F, E), (I, J, K, L).
        #   
        #     A is the implicit first item in the list.
        #     Set front to False to make A the implicit last item in the list.
        #     Set middle to True to return lines where A is in the middle e.g. (H, G, F, E) and (I, J, K, L).
        #     """
        #     answer = []
        #     for l in atom_lines_uniq:
        #         if l[0] == a:
        #             answer.append(l[:][1:])
        #         elif l[-1] == a:
        #             answer.append(l[::-1][1:])
        #         elif middle and a in l:
        #             answer.append(l[l.index(a):][1:])
        #             answer.append(l[:l.index(a)][::-1])
        #     if front: return answer
        #     else: return [l[::-1] for l in answer]
        #
        # def same_line(al, bl):
        #     for l in atom_lines_uniq:
        #         if set(al).issubset(set(l)) and set(bl).issubset(set(l)):
        #             return True
        #     return False
        #
        # ## Multiple angle code; does not improve performance for Fe4N system.
        # for b in molecule.topology.nodes():
        #     for al in pull_lines(b, front=False, middle=True):
        #         for cl in pull_lines(b, front=True, middle=True):
        #             if al[0] == cl[-1]: continue
        #             if al[-1] == cl[0]: continue
        #             self.delete(Angle(al[-1], b, cl[0]))
        #             self.delete(Angle(cl[0], b, al[-1]))
        #             if len(set(al).intersection(set(cl))) > 0: continue
        #             if same_line(al, cl):
        #                 continue
        #             if al[-1] < cl[0]:
        #                 self.add(MultiAngle(al, b, cl))
        #             else:
        #                 self.add(MultiAngle(cl[::-1], b, al[::-1]))
        #
        ## Multiple dihedral code
        ## Note: This suffers from a problem where it cannot rebuild the Cartesian coordinates,
        ## possibly due to a bug in the MultiDihedral class.
        # for aline in atom_lines_uniq:
        #     for (b, c) in itertools.combinations(aline, 2):
        #         if b > c: (b, c) = (c, b)
        #         for al in pull_lines(b, front=False, middle=True):
        #             if same_line(al, aline): continue
        #                 # print "Same line:", al, aline
        #             for dl in pull_lines(c, front=True, middle=True):
        #                 if same_line(dl, aline): continue
        #                     # print "Same line:", dl, aline
        #                     # continue
        #                 # if same_line(dl, al): continue
        #                 if al[-1] == dl[0]: continue
        #                 # if len(set(al).intersection(set(dl))) > 0: continue
        #                 # print MultiDihedral(al, b, c, dl)
        #                 self.delete(Dihedral(al[-1], b, c, dl[0]))
        #                 self.add(MultiDihedral(al, b, c, dl))

    def makeConstraints(self, molecule, constraints, cvals):
        # Add the list of constraints. 
        xyz = molecule.xyzs[0].flatten() * ang2bohr
        if constraints is not None:
            if len(constraints) != len(cvals):
                raise RuntimeError("List of constraints should be same length as constraint values")
            for cons, cval in zip(constraints, cvals):
                self.addConstraint(cons, cval, xyz)

    def __repr__(self):
        lines = ["Internal coordinate system (atoms numbered from 1):"]
        typedict = OrderedDict()
        for Internal in self.Internals:
            lines.append(Internal.__repr__())
            if str(type(Internal)) not in typedict:
                typedict[str(type(Internal))] = 1
            else:
                typedict[str(type(Internal))] += 1
        if len(lines) > 1000:
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
                    logger.info("Deleting:" + str(i) + "\n")
                    self.Internals.remove(i)
                    Changed = True
            else:
                i.inactive = 0
        for i in other.Internals:
            if i not in self.Internals:
                logger.info("Adding:  " + str(i) + "\n")
                self.Internals.append(i)
                Changed = True
        return Changed

    def join(self, other):
        Changed = False
        for i in other.Internals:
            if i not in self.Internals:
                logger.info("Adding:  " + str(i) + "\n")
                self.Internals.append(i)
                Changed = True
        return Changed

    def repr_diff(self, other):
        if hasattr(other, 'Prims'):
            output = ['Primitive -> Delocalized']
            otherPrims = other.Prims
        else:
            output = []
            otherPrims = other
        alines = ["-- Added: --"]
        for i in otherPrims.Internals:
            if i not in self.Internals:
                alines.append(i.__repr__())
        dlines = ["-- Deleted: --"]
        for i in self.Internals:
            if i not in otherPrims.Internals:
                dlines.append(i.__repr__())
        if len(alines) > 1:
            output += alines
        if len(dlines) > 1:
            output += dlines
        return '\n'.join(output)

    def resetRotations(self, xyz):
        for Internal in self.Internals:
            if type(Internal) is LinearAngle:
                Internal.reset(xyz)
        for rot in self.Rotators.values():
            rot.reset(xyz)

    def torsionConstraintLinearAngles(self, coords, thre=175):
        """
        Check if a torsion constrained optimization is about to fail
        because three consecutive atoms are nearly linear.
        """
        
        coords = coords.copy().reshape(-1, 3)
        
        def measure_angle_degrees(i, j, k):
            x1 = coords[i]
            x2 = coords[j]
            x3 = coords[k]
            v1 = x1-x2
            v2 = x3-x2
            n = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            angle = np.arccos(n)
            return angle * 180/ np.pi

        linear_torsion_angles = {}
        for Internal in self.cPrims:
            if type(Internal) is Dihedral:
                a, b, c, d = Internal.a, Internal.b, Internal.c, Internal.d
                abc = measure_angle_degrees(a, b, c)
                bcd = measure_angle_degrees(b, c, d)
                if abc > thre:
                    linear_torsion_angles[(a, b, c)] = abc
                elif bcd > thre:
                    linear_torsion_angles[(b, c, d)] = bcd
        return linear_torsion_angles

    def setRegularization(self, xyz):
        regularization_changed = False
        for Internal in self.Internals:
            if type(Internal) is RotationA:
                if Internal.Rotator.set_regularization(xyz):
                    regularization_changed = True
            elif type(Internal) is LinearAngle:
                Internal.reposition_e0(xyz)
        return regularization_changed

    def largeRots(self):
        for Internal in self.Internals:
            if type(Internal) in [RotationA, RotationB, RotationC]:
                if Internal in self.cPrims:
                    continue
                if Internal.Rotator.stored_norm > 0.9*np.pi:
                    # # Molecule has rotated by almost pi
                    if type(Internal) is RotationA:
                        logger.info("Large rotation: %s = %.3f*pi\n" % (str(Internal), Internal.Rotator.stored_norm/np.pi))
                    return True
        return False

    def calculate(self, xyz):
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.value(xyz))
        return np.array(answer)

    def getRotatorNorms(self):
        rots = []
        for Internal in self.Internals:
            if type(Internal) in [RotationA]:
                rots.append(Internal.Rotator.stored_norm)
        return rots

    def printRotations(self, xyz):
        rotNorms = self.getRotatorNorms()
        if len(rotNorms) > 0:
            logger.info("Rotator Norms: " + " ".join(["% .4f" % i for i in rotNorms]) + "\n")
        linAngs = [ic.value(xyz) for ic in self.Internals if type(ic) is LinearAngle]
        if len(linAngs) > 0:
            logger.info("Linear Angles: " + " ".join(["% .4f" % i for i in linAngs]) + "\n")

    def derivatives(self, xyz):
        self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        return np.array(answer)

    def second_derivatives(self, xyz):
        self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.second_derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        # 4) Number of atoms
        # 5) 3
        return np.array(answer)
    
    def calcDiff(self, xyz1, xyz2):
        """ Calculate difference in internal coordinates (coord1-coord2), accounting for changes in 2*pi of angles. """
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.calcDiff(xyz1, xyz2))
        return np.array(answer)
    
    def GInverse(self, xyz):
        return self.GInverse_SVD(xyz)

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
            iPrim = self.cPrims.index(cPrim)
            if np.abs(cVal - self.cVals[iPrim]) > 1e-6:
                logger.info("Updating constraint value to %.4e\n" % cVal)
            self.cVals[iPrim] = cVal
        else:
            if cPrim not in self.Internals:
                self.Internals.append(cPrim)
            self.cPrims.append(cPrim)
            self.cVals.append(cVal)

    def reorderPrimitives(self):
        # Reorder primitives to be in line with cc's code
        newPrims = []
        for cPrim in self.cPrims:
            newPrims.append(cPrim)
        for typ in [Distance, Angle, LinearAngle, MultiAngle, OutOfPlane, Dihedral, MultiDihedral, CartesianX, CartesianY, CartesianZ, TranslationX, TranslationY, TranslationZ, RotationA, RotationB, RotationC]:
            for p in self.Internals:
                if type(p) is typ and p not in self.cPrims:
                    newPrims.append(p)
        if len(newPrims) != len(self.Internals):
            raise RuntimeError("Not all internal coordinates have been accounted for. You may need to add something to reorderPrimitives()")
        self.Internals = newPrims

    def getConstraints_from(self, other):
        if other.haveConstraints():
            for cPrim, cVal in zip(other.cPrims, other.cVals):
                self.addConstraint(cPrim, cVal)
        self.reorderPrimitives()

    def haveConstraints(self):
        return len(self.cPrims) > 0

    def getConstraintNames(self):
        return [str(c) for c in self.cPrims]
        
    def getConstraintTargetVals(self, units=False):
        if units:
            return convert_angstroms_degrees(self.cPrims, self.cVals)
        else:
            return self.cVals

    def getConstraintCurrentVals(self, xyz, units=False):
        answer = []
        for ic, c in enumerate(self.cPrims):
            value = c.value(xyz)
            answer.append(value)
        if units:
            return convert_angstroms_degrees(self.cPrims, np.array(answer))
        else:
            return np.array(answer)

    def calcConstraintDiff(self, xyz, units=False):
        """ Calculate difference between 
        (constraint ICs evaluated at provided coordinates - constraint values).

        If units=True then the values will be returned in units of Angstrom and degrees 
        for distance and angle degrees of freedom respectively.
        """
        cDiffs = np.zeros(len(self.cPrims), dtype=float)
        for ic, c in enumerate(self.cPrims):
            # Calculate the further change needed in this constrained variable
            if type(c) is RotationA:
                ca = c
                cb = self.cPrims[ic+1]
                cc = self.cPrims[ic+2]
                if type(cb) is not RotationB or type(cc) is not RotationC:
                    raise RuntimeError('In primitive internal coordinates, RotationA must be followed by RotationB and RotationC.')
                if len(set([ca.w, cb.w, cc.w])) != 1:
                    raise RuntimeError('The triple of rotation ICs need to have the same weight.')
                cDiffs[ic] = ca.calcDiff(xyz, val2=self.cVals[ic:ic+3]/c.w)
                cDiffs[ic+1] = cb.calcDiff(xyz, val2=self.cVals[ic:ic+3]/c.w)
                cDiffs[ic+2] = cc.calcDiff(xyz, val2=self.cVals[ic:ic+3]/c.w)
            elif type(c) in [RotationB, RotationC]: pass
            else:
                cDiffs[ic] = c.calcDiff(xyz, val2=self.cVals[ic])
        if units:
            return convert_angstroms_degrees(self.cPrims, cDiffs)
        else:
            return cDiffs
    
    def maxConstraintViolation(self, xyz):
        cDiffs = self.calcConstraintDiff(xyz, units=True)
        return np.max(np.abs(cDiffs))

    def printConstraints(self, xyz, thre=1e-5):
        nc = len(self.cPrims)
        out_lines = []
        header = "Constraint                         Current      Target       Diff."
        curr = self.getConstraintCurrentVals(xyz, units=True)
        refs = self.getConstraintTargetVals(units=True)
        diff = self.calcConstraintDiff(xyz, units=True)
        for ic, c in enumerate(self.cPrims):
            if np.abs(diff[ic]) > thre:
                out_lines.append("%-30s  % 10.5f  % 10.5f  % 10.5f" % (str(c), curr[ic], refs[ic], diff[ic]))
        if len(out_lines) > 0:
            logger.info(header + "\n")
            logger.info('\n'.join(out_lines) + "\n")

    def visualizeRotations(self, xyz):
        rxyz = []
        relem = []
        ridx = []
        elem = np.array(self.elem, dtype=object)
        for ic in self.Internals:
            if type(ic) is RotationA:
                rxyz.append(ic.Rotator.visualize(xyz))
                ridx.append(ic.Rotator.a[-1]+1)
                relem.append(np.array(['X', 'X'], dtype=object))
            elif type(ic) is LinearAngle and ic.axis == 0:
                rxyz.append(ic.visualize(xyz))
                ridx.append(ic.c+1)
                relem.append(np.array(['Z', 'Z', 'Z'], dtype=object))
                # relem.append('He')
                # relem.append('He')

        rsort = np.argsort(ridx)
        xyz_with_r = xyz.reshape(-1, 3).copy()
        elem_with_r = elem.copy()
        for i in range(len(ridx)):
            j=len(ridx)-i-1
            xyz_with_r = np.insert(xyz_with_r, ridx[rsort[j]], rxyz[rsort[j]], axis=0)
            elem_with_r = np.insert(elem_with_r, ridx[rsort[j]], relem[rsort[j]])

        # rxyz = np.array(rxyz)
        # ridx = np.array(ridx)
        # relem = np.array(relem)
        # xyz_with_r = np.insert(xyz.copy().reshape(-1,3), ridx, rxyz)
        # elem_with_r = np.insert(elem, ridx, relem)
        M = Molecule()
        M.xyzs = [xyz_with_r*bohr2ang]
        M.elem = list(elem_with_r)
        return M
            
    def guess_hessian(self, coords):
        """
        Build a guess Hessian that roughly follows Schlegel's guidelines. 
        """
        xyzs = coords.reshape(-1,3)*bohr2ang
        Hdiag = []
        def covalent(a, b):
            r = np.linalg.norm(xyzs[a]-xyzs[b])
            rcov = Radii[Elements.index(self.elem[a])-1] + Radii[Elements.index(self.elem[b])-1]
            return r/rcov < 1.2
        
        for ic in self.Internals:
            if type(ic) is Distance:
                r = np.linalg.norm(xyzs[ic.a]-xyzs[ic.b]) * ang2bohr
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
            elif type(ic) in [Angle, LinearAngle, MultiAngle]:
                if type(ic) in [Angle, LinearAngle]:
                    a = ic.a
                    c = ic.c
                else:
                    a = ic.a[-1]
                    c = ic.c[0]
                if min(Elements.index(self.elem[a]),
                       Elements.index(self.elem[ic.b]),
                       Elements.index(self.elem[c])) < 3:
                    A = 0.160
                else:
                    A = 0.250
                if covalent(a, ic.b) and covalent(ic.b, c):
                    Hdiag.append(A)
                else:
                    Hdiag.append(0.1)
            elif type(ic) in [Dihedral, MultiDihedral]:
                r = (np.linalg.norm(xyzs[ic.b]-xyzs[ic.c]))*ang2bohr
                rcov = (Radii[Elements.index(self.elem[ic.b])-1] + Radii[Elements.index(self.elem[ic.c])-1])*ang2bohr
                #Hdiag.append(min(0.023, 0.005 - 0.07*min(0.0, r-rcov)))
                Hdiag.append(0.023)
            elif type(ic) is OutOfPlane:
                r1 = xyzs[ic.b]-xyzs[ic.a]
                r2 = xyzs[ic.c]-xyzs[ic.a]
                r3 = xyzs[ic.d]-xyzs[ic.a]
                d = 1 - np.abs(np.dot(r1,np.cross(r2,r3))/np.linalg.norm(r1)/np.linalg.norm(r2)/np.linalg.norm(r3))
                # Hdiag.append(0.1)
                if covalent(ic.a, ic.b) and covalent(ic.a, ic.c) and covalent(ic.a, ic.d):
                    Hdiag.append(0.045)
                else:
                    Hdiag.append(0.023)
            elif type(ic) in [CartesianX, CartesianY, CartesianZ]:
                Hdiag.append(0.05)
            elif type(ic) in [TranslationX, TranslationY, TranslationZ]:
                Hdiag.append(0.05)
            elif type(ic) in [RotationA, RotationB, RotationC]:
                Hdiag.append(0.05)
            else:
                raise RuntimeError('Failed to build guess Hessian matrix. Make sure all IC types are supported')
        return np.diag(Hdiag)

    
class DelocalizedInternalCoordinates(InternalCoordinates):
    def __init__(self, molecule, imagenr=0, build=False, connect=False, addcart=False, constraints=None, cvals=None, remove_tr=False, cart_only=False, conmethod=0):
        super(DelocalizedInternalCoordinates, self).__init__()
        # cart_only is just because of how I set up the class structure.
        if cart_only: return
        # Set the algorithm for constraint satisfaction.
        # 0 - Original algorithm implemented in 2016, constraints are satisfied slowly unless "enforce" is enabled
        # 1 - Updated algorithm implemented on 2019-03-20, constraints are satisfied instantly, "enforce" is not needed
        self.conmethod = conmethod
        # HDLC is given by (connect = False, addcart = True)
        # Standard DLC is given by (connect = True, addcart = False)
        # TRIC is given by (connect = False, addcart = False)
        # Build a minimum spanning tree 
        self.connect = connect
        # Add Cartesian coordinates to all.
        self.addcart = addcart
        # The DLC contains an instance of primitive internal coordinates.
        self.Prims = PrimitiveInternalCoordinates(molecule, connect=connect, addcart=addcart, constraints=constraints, cvals=cvals)
        self.frags = self.Prims.frags
        self.na = molecule.na
        # Whether constraints have been enforced previously
        self.enforced = False
        self.enforce_fail_printed = False
        # Build the DLC's. This takes some time, so we have the option to turn it off.
        xyz = molecule.xyzs[imagenr].flatten() * ang2bohr
        if build:
            self.build_dlc(xyz)
        self.remove_tr = remove_tr
        if self.remove_tr:
            self.remove_TR(xyz)

    def clearCache(self):
        super(DelocalizedInternalCoordinates, self).clearCache()
        self.Prims.clearCache()

    def __repr__(self):
        return self.Prims.__repr__()
            
    def update(self, other):
        return self.Prims.update(other.Prims)
        
    def join(self, other):
        return self.Prims.join(other.Prims)
        
    def addConstraint(self, cPrim, cVal, xyz):
        self.Prims.addConstraint(cPrim, cVal, xyz)

    def getConstraints_from(self, other):
        self.Prims.getConstraints_from(other.Prims)
        
    def haveConstraints(self):
        return len(self.Prims.cPrims) > 0

    def getConstraintNames(self):
        return self.Prims.getConstraintNames()
    
    def getConstraintTargetVals(self, units=True):
        return self.Prims.getConstraintTargetVals(units=units)
    
    def getConstraintCurrentVals(self, xyz, units=True):
        return self.Prims.getConstraintCurrentVals(xyz, units=units)

    def calcConstraintDiff(self, xyz, units=False):
        return self.Prims.calcConstraintDiff(xyz, units=units)
    
    def maxConstraintViolation(self, xyz):
        return self.Prims.maxConstraintViolation(xyz)

    def printConstraints(self, xyz, thre=1e-5):
        self.Prims.printConstraints(xyz, thre=thre)

    def visualizeRotations(self, xyz):
        return self.Prims.visualizeRotations(xyz)
    
    def augmentGH(self, xyz, G, H):
        """
        Add extra dimensions to the gradient and Hessian corresponding to the constrained degrees of freedom.
        The Hessian becomes:  H  c
                              cT 0
        where the elements of cT are the first derivatives of the constraint function 
        (typically a single primitive minus a constant) with respect to the DLCs. 
        
        Since we picked a DLC to represent the constraint (cProj), we only set one element 
        in each row of cT to be nonzero. Because cProj = a_i * Prim_i + a_j * Prim_j, we have
        d(Prim_c)/d(cProj) = 1.0/a_c where "c" is the index of the primitive being constrained.
        
        The extended elements of the Gradient are equal to the constraint violation.
        
        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        G : np.ndarray
            Flat array containing internal coordinate gradient
        H : np.ndarray
            Square array containing internal coordinate Hessian

        Returns
        -------
        GC : np.ndarray
            Flat array containing gradient extended by constraint violations
        HC : np.ndarray
            Square matrix extended by partial derivatives d(Prim)/d(cProj)
        """
        # Number of internals (elements of G)
        ni = len(G)
        # Number of constraints
        nc = len(self.Prims.cPrims)
        # Total dimension
        nt = ni+nc
        # Lower block of the augmented Hessian
        cT = np.zeros((nc, ni), dtype=float)
        # The further change needed in constrained variables:
        # (Constraint values) - (Current values of constraint ICs)
        c0 = -1.0 * self.calcConstraintDiff(xyz)
        for ic, c in enumerate(self.Prims.cPrims):
            # Look up the index of the primitive that is being constrained
            iPrim = self.Prims.Internals.index(c)
            # The DLC corresponding to the constrained primitive (a.k.a. cProj) is self.Vecs[self.cDLC[ic]].
            # For a differential change in the DLC, the primitive that we are constraining changes by:
            cT[ic, self.cDLC[ic]] = 1.0/self.Vecs[iPrim, self.cDLC[ic]]
            # The new constraint algorithm satisfies constraints too quickly and could cause
            # the energy to blow up. Thus, constraint steps are restricted to 0.1 au/radian
            if self.conmethod == 1:
                if c0[ic] < -0.1:
                    c0[ic] = -0.1
                if c0[ic] > 0.1:
                    c0[ic] = 0.1
        # Construct augmented Hessian
        HC = np.zeros((nt, nt), dtype=float)
        HC[0:ni, 0:ni] = H[:,:]
        HC[ni:nt, 0:ni] = cT[:,:]
        HC[0:ni, ni:nt] = cT.T[:,:]
        # Construct augmented gradient
        GC = np.zeros(nt, dtype=float)
        GC[0:ni] = G[:]
        GC[ni:nt] = -c0[:]
        return GC, HC
    
    def applyConstraints(self, xyz):
        """
        Pass in Cartesian coordinates and return new coordinates that satisfy the constraints exactly.
        """
        xyz1 = xyz.copy()
        niter = 0
        xyzs = []
        ndqs = []
        while True:
            dQ = np.zeros(len(self.Internals), dtype=float)
            cDiff = -1.0 * self.calcConstraintDiff(xyz1)
            for ic, c in enumerate(self.Prims.cPrims):
                # Look up the index of the primitive that is being constrained
                iPrim = self.Prims.Internals.index(c)
                # Look up the index of the DLC that corresponds to the constraint
                iDLC = self.cDLC[ic]
                # Calculate the further change needed in this constrained variable
                dQ[iDLC] = cDiff[ic]
                dQ[iDLC] /= self.Vecs[iPrim, iDLC]
            xyzs.append(xyz1.copy())
            ndqs.append(np.linalg.norm(dQ))
            # print("applyConstraints calling newCartesian (%i), |dQ| = %.3e" % (niter, np.linalg.norm(dQ)))
            xyz2 = self.newCartesian(xyz1, dQ, verbose=0)
            if np.linalg.norm(dQ) < 1e-6:
                return xyz2
            if niter > 1 and np.linalg.norm(dQ) > np.linalg.norm(dQ0):
                xyz1 = xyzs[np.argmin(ndqs)]
                if not self.enforce_fail_printed:
                    logger.warning("Warning: Failed to enforce exact constraint satisfaction. Please remove possible redundant constraints. See below:\n")
                    self.printConstraints(xyz1, thre=0.0)
                    self.enforce_fail_printed = True
                return xyz1
            xyz1 = xyz2.copy()
            niter += 1
            dQ0 = dQ.copy()
            
    def newCartesian_withConstraint(self, xyz, dQ, thre=0.1, verbose=0):
        xyz2 = self.newCartesian(xyz, dQ, verbose)
        constraintSmall = len(self.Prims.cPrims) > 0
        cDiff = self.calcConstraintDiff(xyz)
        for ic, c in enumerate(self.Prims.cPrims):
            diff = cDiff[ic]
            if np.abs(diff) > thre:
                constraintSmall = False
        if constraintSmall:
            xyz2 = self.applyConstraints(xyz2)
            if not self.enforced:
                logger.info("<<< Enforcing constraint satisfaction >>>\n")
            self.enforced = True
        else:
            self.enforced = False
        return xyz2
    
    def calcGradProj(self, xyz, gradx):
        """
        Project out the components of the internal coordinate gradient along the
        constrained degrees of freedom. This is used to calculate the convergence
        criteria for constrained optimizations.

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        gradx : np.ndarray
            Flat array containing gradient in Cartesian coordinates

        Returns
        -------
        np.ndarray
            Flat array containing gradient in Cartesian coordinates with forces
            along constrained directions projected out
        """
        if len(self.Prims.cPrims) == 0:
            return gradx
        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
        Gq = multi_dot([Ginv, Bmat, gradx.T])
        Gqc = np.array(Gq).flatten()
        # Remove the directions that are along the DLCs that we are constraining
        for i in self.cDLC:
            Gqc[i] = 0.0
        # Gxc = np.array(np.matrix(Bmat.T)*np.matrix(Gqc).T).flatten()
        Gxc = multi_dot([Bmat.T, Gqc.T]).flatten()
        return Gxc
    
    def build_dlc_0(self, xyz):
        """
        Build the delocalized internal coordinates (DLCs) which are linear 
        combinations of the primitive internal coordinates. Each DLC is stored
        as a column in self.Vecs.
        
        In short, each DLC is an eigenvector of the G-matrix, and the number of
        nonzero eigenvalues of G should be equal to 3*N. 
        
        After creating the DLCs, we construct special ones corresponding to primitive
        coordinates that are constrained (cProj).  These are placed in the front (i.e. left)
        of the list of DLCs, and then we perform a Gram-Schmidt orthogonalization.

        This function is called at the end of __init__ after the coordinate system is already
        specified (including which primitives are constraints).

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        """
        # Perform singular value decomposition
        click()
        G = self.Prims.GMatrix(xyz)
        # Manipulate G-Matrix to increase weight of constrained coordinates
        if self.haveConstraints():
            for ic, c in enumerate(self.Prims.cPrims):
                iPrim = self.Prims.Internals.index(c)
                G[:, iPrim] *= 1.0
                G[iPrim, :] *= 1.0
        ncon = len(self.Prims.cPrims)
        # Water Dimer: 100.0, no check -> -151.1892668451
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
        Expect = 3*self.na
        # print "%i atoms (expect %i coordinates); %i/%i singular values are > 1e-6" % (self.na, Expect, LargeVals, len(L))
        # if LargeVals <= Expect:
        self.Vecs = Q[:, LargeIdx]

        # Vecs has number of rows equal to the number of primitives, and
        # number of columns equal to the number of delocalized internal coordinates.
        if self.haveConstraints():
            click()
            # print "Projecting out constraints...",
            V = []
            for ic, c in enumerate(self.Prims.cPrims):
                # Look up the index of the primitive that is being constrained
                iPrim = self.Prims.Internals.index(c)
                # Pick a row out of the eigenvector space. This is a linear combination of the DLCs.
                cVec = self.Vecs[iPrim, :]
                cVec = np.array(cVec)
                cVec /= np.linalg.norm(cVec)
                # This is a "new DLC" that corresponds to the primitive that we are constraining
                cProj = np.dot(self.Vecs,cVec.T)
                cProj /= np.linalg.norm(cProj)
                V.append(np.array(cProj).flatten())
                # print c, cProj[iPrim]
            # V contains the constraint vectors on the left, and the original DLCs on the right
            V = np.hstack((np.array(V).T, np.array(self.Vecs)))
            # Apply Gram-Schmidt to V, and produce U.
            # The Gram-Schmidt process should produce a number of orthogonal DLCs equal to the original number
            thre = 1e-6
            while True:
                U = []
                for iv in range(V.shape[1]):
                    v = V[:, iv].flatten()
                    U.append(v.copy())
                    for ui in U[:-1]:
                        U[-1] -= ui * np.dot(ui, v)
                    if np.linalg.norm(U[-1]) < thre:
                        U = U[:-1]
                        continue
                    U[-1] /= np.linalg.norm(U[-1])
                if len(U) > self.Vecs.shape[1]:
                    thre *= 10
                elif len(U) == self.Vecs.shape[1]:
                    break
                elif len(U) < self.Vecs.shape[1]:
                    raise RuntimeError('Gram-Schmidt orthogonalization has failed (expect %i length %i)' % (self.Vecs.shape[1], len(U)))
            # print "Gram-Schmidt completed with thre=%.0e" % thre
            self.Vecs = np.array(U).T
            # Constrained DLCs are on the left of self.Vecs.
            self.cDLC = [i for i in range(len(self.Prims.cPrims))]
        # Now self.Internals is no longer a list of InternalCoordinate objects but only a list of strings.
        # We do not create objects for individual DLCs but 
        self.Internals = ["Constraint-DLC" if i < ncon else "DLC" + " %i" % (i+1) for i in range(self.Vecs.shape[1])]

    def build_dlc_1(self, xyz):
        """
        Build the delocalized internal coordinates (DLCs) which are linear 
        combinations of the primitive internal coordinates. Each DLC is stored
        as a column in self.Vecs.
        
        After some thought, build_dlc_0 did not implement constraint satisfaction
        in the most correct way. Constraint satisfaction was rather slow and
        the --enforce 0.1 may be passed to improve performance. Rethinking how the
        G matrix is constructed provides a more fundamental solution.

        In the new approach implemented here, constrained primitive ICs (PICs) are 
        first set aside from the rest of the PICs. Next, a G-matrix is constructed
        from the rest of the PICs and diagonalized to form DLCs, called "residual" DLCs. 
        The union of the cPICs and rDLCs forms a generalized set of DLCs, but the
        cPICs are not orthogonal to each other or to the rDLCs.
        
        A set of orthogonal DLCs is constructed by carrying out Gram-Schmidt
        on the generalized set. Orthogonalization is carried out on the cPICs in order.
        Next, orthogonalization is carried out on the rDLCs using a greedy algorithm
        that carries out projection for each cPIC, then keeps the one with the largest
        remaining norm. This way we avoid keeping rDLCs that are almost redundant with
        the cPICs. The longest projected rDLC is added to the set and continued until
        the expected number is reached.

        One special note in orthogonalization is that the "overlap" between internal
        coordinates corresponds to the G matrix element. Thus, for DLCs that's a linear
        combination of PICs, then the overlap is given by:

        v_i * B * B^T * v_j = v_i * G * v_j

        Notes on usage: 1) When this is activated, constraints tend to be satisfied very 
        rapidly even if the current coordinates are very far from the constraint values,
        leading to possible blowing up of the energies. In augment_GH, maximum steps in
        constrained degrees of freedom are restricted to 0.1 a.u./radian for this reason.
        
        2) Although the performance of this method is generally superior to the old method,
        the old method with --enforce 0.1 is more extensively tested and recommended.
        Thus, this method isn't enabled by default but provided as an optional feature.

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        """
        click()
        G = self.Prims.GMatrix(xyz)
        nprim = len(self.Prims.Internals)
        cPrimIdx = []
        if self.haveConstraints():
            for ic, c in enumerate(self.Prims.cPrims):
                iPrim = self.Prims.Internals.index(c)
                cPrimIdx.append(iPrim)
        ncon = len(self.Prims.cPrims)
        if cPrimIdx != list(range(ncon)):
            raise RuntimeError("The constraint primitives should be at the start of the list")
        # Form a sub-G-matrix that doesn't include the constrained primitives and diagonalize it to form DLCs.
        Gsub = G[ncon:, ncon:]
        time_G = click()
        L, Q = np.linalg.eigh(Gsub)
        # Sort eigenvalues and eigenvectors in descending order (for cleanliness)
        L = L[::-1]
        Q = Q[:, ::-1]
        time_eig = click()
        # print "Build G: %.3f Eig: %.3f" % (time_G, time_eig)
        # Figure out which eigenvectors from the G submatrix to include
        LargeVals = 0
        LargeIdx = []
        GEigThre = 1e-6
        for ival, value in enumerate(L):
            if np.abs(value) > GEigThre:
                LargeVals += 1
                LargeIdx.append(ival)
        # This is the number of nonredundant DLCs that we expect to have at the end
        Expect = np.sum(np.linalg.eigh(G)[0] > 1e-6)

        if (ncon + len(LargeIdx)) < Expect:
            raise RuntimeError("Expected at least %i delocalized coordinates, but got only %i" % (Expect, ncon + len(LargeIdx)))
        # print("%i atoms (expect %i coordinates); %i/%i singular values are > 1e-6" % (self.na, Expect, LargeVals, len(L)))
        
        # Create "generalized" DLCs where the first six columns are the constrained primitive ICs
        # and the other columns are the DLCs formed from the rest
        self.Vecs = np.zeros((nprim, ncon+LargeVals), dtype=float)
        for i in range(ncon):
            self.Vecs[i, i] = 1.0
        self.Vecs[ncon:, ncon:ncon+LargeVals] = Q[:, LargeIdx]

        # Perform Gram-Schmidt orthogonalization
        def ov(vi, vj):
            return multi_dot([vi, G, vj])
        if self.haveConstraints():
            click()
            V = self.Vecs.copy()
            nv = V.shape[1]
            Vnorms = np.array([np.sqrt(ov(V[:,ic], V[:, ic])) for ic in range(nv)])
            # U holds the Gram-Schmidt orthogonalized DLCs
            U = np.zeros((V.shape[0], Expect), dtype=float)
            Unorms = np.zeros(Expect, dtype=float)
            
            for ic in range(ncon):
                # At the top of the loop, V columns are orthogonal to U columns up to ic.
                # Copy V column corresponding to the next constraint to U.
                U[:, ic] = V[:, ic].copy()
                ui = U[:, ic]
                Unorms[ic] = np.sqrt(ov(ui, ui))
                if Unorms[ic]/Vnorms[ic] < 0.1:
                    logger.warning("Constraint %i is almost redundant; after projection norm is %.3f of original\n" % (ic, Unorms[ic]/Vnorms[ic]))
                V0 = V.copy()
                # Project out newest U column from all remaining V columns.
                for jc in range(ic+1, nv):
                    vj = V[:, jc]
                    vj -= ui * ov(ui, vj)/Unorms[ic]**2
                
            for ic in range(ncon, Expect):
                # Pick out the V column with the largest norm
                norms = np.array([np.sqrt(ov(V[:, jc], V[:, jc])) for jc in range(ncon, nv)])
                imax = ncon+np.argmax(norms)
                # Add this column to U
                U[:, ic] = V[:, imax].copy()
                ui = U[:, ic]
                Unorms[ic] = np.sqrt(ov(ui, ui))
                # Project out the newest U column from all V columns
                for jc in range(ncon, nv):
                    V[:, jc] -= ui * ov(ui, V[:, jc])/Unorms[ic]**2
                
            # self.Vecs contains the linear combination coefficients that are our new DLCs
            self.Vecs = U.copy()
            # Constrained DLCs are on the left of self.Vecs.
            self.cDLC = [i for i in range(len(self.Prims.cPrims))]

        self.Internals = ["Constraint" if i < ncon else "DLC" + " %i" % (i+1) for i in range(self.Vecs.shape[1])]
        # # LPW: Coefficients of DLC's are in each column and DLCs corresponding to constraints should basically be like (0 1 0 0 0 ..)
        # pmat2d(self.Vecs, format='f', precision=2)
        # B = self.Prims.wilsonB(xyz)
        # Bdlc = np.einsum('ji,jk->ik', self.Vecs, B)
        # Gdlc = np.dot(Bdlc, Bdlc.T)
        # # Expect to see a diagonal matrix here
        # print("Gdlc")
        # pmat2d(Gdlc, format='e', precision=2)
        # # Expect to see "large" eigenvalues here (no less than 0.1 ideally)
        # print("L, Q")
        # L, Q = np.linalg.eigh(Gdlc)
        # print(L)

    def build_dlc(self, xyz):
        if self.conmethod == 1:
            return self.build_dlc_1(xyz)
        elif self.conmethod == 0:
            return self.build_dlc_0(xyz)
        else:
            raise RuntimeError("Unsupported value of conmethod %i" % self.conmethod)
            
    def remove_TR(self, xyz):
        """
        Project overall translation and rotation out of the DLCs.
        This feature is intended to be used when an optimization job appears
        to contain slow rotations of the whole system, which sometimes happens.
        Uses the same logic as build_dlc_1.
        """
        # Create three translation and three rotation primitive ICs for the whole system
        na = int(len(xyz)/3)
        alla = range(na)
        sel = xyz.reshape(-1,3).copy()
        TRPrims = []
        TRPrims.append(TranslationX(alla, w=np.ones(na)/na))
        TRPrims.append(TranslationY(alla, w=np.ones(na)/na))
        TRPrims.append(TranslationZ(alla, w=np.ones(na)/na))
        sel -= np.mean(sel, axis=0)
        rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
        TRPrims.append(RotationA(alla, xyz, self.Prims.Rotators, w=rg))
        TRPrims.append(RotationB(alla, xyz, self.Prims.Rotators, w=rg))
        TRPrims.append(RotationC(alla, xyz, self.Prims.Rotators, w=rg))
        # If these primitive ICs are already there, then move them to the front
        primorder = []
        addPrims = []
        for prim in TRPrims:
            if prim in self.Prims.Internals:
                primorder.append(self.Prims.Internals.index(prim))
            else:
                addPrims.append(prim)
        for iprim, prim in enumerate(self.Prims.Internals):
            if prim not in TRPrims:
                primorder.append(iprim)
        self.Prims.Internals = addPrims + [self.Prims.Internals[p] for p in primorder]
        self.Vecs = np.vstack((np.zeros((len(addPrims), self.Vecs.shape[1]), dtype=float), self.Vecs[np.array(primorder), :]))
        
        self.clearCache()
        # Build DLCs with six extra in the front corresponding to the overall translations and rotations
        subVecs = self.Vecs.copy()
        self.Vecs = np.zeros((self.Vecs.shape[0], self.Vecs.shape[1]+6), dtype=float)
        self.Vecs[:6, :6] = np.eye(6)
        self.Vecs[:, 6:] = subVecs.copy()
        # pmat2d(self.Vecs, precision=3, format='f')
        # sys.exit()
        # This is the number of nonredundant DLCs that we expect to see
        G = self.Prims.GMatrix(xyz)
        Expect = np.sum(np.linalg.eigh(G)[0] > 1e-6)

        # Carry out Gram-Schmidt orthogonalization
        # Define a function for computing overlap
        def ov(vi, vj):
            return multi_dot([vi, G, vj])
        V = self.Vecs.copy()
        nv = V.shape[1]
        Vnorms = np.array([np.sqrt(ov(V[:, ic], V[:, ic])) for ic in range(nv)])
        # G_tmp = np.array([[ov(V[:, ic], V[:, jc]) for jc in range(nv)] for ic in range(nv)])
        # pmat2d(G_tmp, precision=3, format='f')
        # U holds the Gram-Schmidt orthogonalized DLCs
        U = np.zeros((V.shape[0], Expect), dtype=float)
        Unorms = np.zeros(Expect, dtype=float)
        ncon = len(self.Prims.cPrims)
        # Keep translations, rotations, and any constraints in sequential order
        # and project them out of the remaining DLCs
        for ic in range(6+ncon):
            U[:, ic] = V[:, ic].copy()
            ui = U[:, ic]
            Unorms[ic] = np.sqrt(ov(ui, ui))
            if Unorms[ic]/Vnorms[ic] < 0.1:
                logger.warning("Constraint %i is almost redundant; after projection norm is %.3f of original\n" % (ic-6, Unorms[ic]/Vnorms[ic]))
            V0 = V.copy()
            # Project out newest U column from all remaining V columns.
            for jc in range(ic+1, nv):
                vj = V[:, jc]
                vj -= ui * ov(ui, vj)/Unorms[ic]**2
        # Now keep the remaining DLC with the largest norm, perform projection,
        # then repeat until the expected number is found
        shift = 6+ncon
        for ic in range(shift, Expect):
            # Pick out the V column with the largest norm
            norms = np.array([np.sqrt(ov(V[:, jc], V[:, jc])) for jc in range(shift, nv)])
            imax = shift+np.argmax(norms)
            # Add this column to U
            U[:, ic] = V[:, imax].copy()
            ui = U[:, ic]
            Unorms[ic] = np.sqrt(ov(ui, ui))
            # Project out the newest U column from all V columns
            for jc in range(ncon, nv):
                V[:, jc] -= ui * ov(ui, V[:, jc])/Unorms[ic]**2
        # self.Vecs contains the linear combination coefficients that are our new DLCs
        self.Vecs = U[:, 6:].copy()
        self.Internals = ["Constraint" if i < ncon else "DLC" + " %i" % (i+1) for i in range(self.Vecs.shape[1])]

    def weight_vectors(self, xyz):
        """
        Not used anymore: Multiply each DLC by a constant so that a small displacement along each produces the
        same Cartesian displacement. Otherwise, some DLCs "move by a lot" and others only "move by a little".

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        """
        Bmat = self.wilsonB(xyz)
        Ginv = self.GInverse(xyz, None)
        eps = 1e-6
        dxdq = np.zeros(len(self.Internals))
        for i in range(len(self.Internals)):
            dQ = np.zeros(len(self.Internals), dtype=float)
            dQ[i] = eps
            dxyz = multi_dot([Bmat.T, Ginv , dQ.T])
            rmsd = np.sqrt(np.mean(np.sum(np.array(dxyz).reshape(-1,3)**2, axis=1)))
            dxdq[i] = rmsd/eps
        dxdq /= np.max(dxdq)
        for i in range(len(self.Internals)):
            self.Vecs[:, i] *= dxdq[i]

    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)

    def torsionConstraintLinearAngles(self, coords, thre=175):
        """ Check if certain problems might be happening due to three consecutive atoms in a torsion angle becoming linear. """
        return self.Prims.torsionConstraintLinearAngles(coords, thre)
    
    def setRegularization(self, xyz):
        return self.Prims.setRegularization(xyz)

    def largeRots(self):
        """ Determine whether a molecule has rotated by an amount larger than some threshold (hardcoded in Prims.largeRots()). """
        return self.Prims.largeRots()

    def printRotations(self, xyz):
        return self.Prims.printRotations(xyz)

    def calcDiff(self, coord1, coord2):
        """ Calculate difference in internal coordinates (coord1-coord2), accounting for changes in 2*pi of angles. """
        PMDiff = self.Prims.calcDiff(coord1, coord2)
        Answer = np.dot(PMDiff, self.Vecs)
        return np.array(Answer).flatten()

    def calculate(self, coords):
        """ Calculate the DLCs given the Cartesian coordinates. """
        PrimVals = self.Prims.calculate(coords)
        Answer = np.dot(PrimVals, self.Vecs)
        # To obtain the primitive coordinates from the delocalized internal coordinates,
        # simply multiply self.Vecs*Answer.T where Answer.T is the column vector of delocalized
        # internal coordinates. That means the "c's" in Equation 23 of Schlegel's review paper
        # are simply the rows of the Vecs matrix.
        # print np.dot(np.array(self.Vecs[0,:]).flatten(), np.array(Answer).flatten())
        # print PrimVals[0]
        # raw_input()
        return np.array(Answer).flatten()

    def derivatives(self, coords):
        """ Obtain the change of the DLCs with respect to the Cartesian coordinates. """
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

    def second_derivatives(self, coords):
        """ Obtain the second derivatives of the DLCs with respect to the Cartesian coordinates. """
        PrimDers = self.Prims.second_derivatives(coords)
        Answer2 = np.tensordot(self.Vecs, PrimDers, axes=(0, 0))
        return np.array(Answer2)
    
    def GInverse(self, xyz):
        return self.GInverse_SVD(xyz)

    def repr_diff(self, other):
        if hasattr(other, 'Prims'):
            return self.Prims.repr_diff(other.Prims)
        else:
            if self.Prims.repr_diff(other) == '':
                return 'Delocalized -> Primitive'
            else:
                return 'Delocalized -> Primitive\n' + self.Prims.repr_diff(other)

    def guess_hessian(self, coords):
        """ Build the guess Hessian, consisting of a diagonal matrix 
        in the primitive space and changed to the basis of DLCs. """
        Hprim = self.Prims.guess_hessian(coords)
        return multi_dot([self.Vecs.T,Hprim,self.Vecs])

    def resetRotations(self, xyz):
        """ Reset the reference geometries for calculating the orientational variables. """
        self.Prims.resetRotations(xyz)

class CartesianCoordinates(PrimitiveInternalCoordinates):
    """
    Cartesian coordinate system, written as a kind of internal coordinate class.  
    This one does not support constraints, because that requires adding some 
    primitive internal coordinates.
    """
    def __init__(self, molecule, **kwargs):
        super(CartesianCoordinates, self).__init__(molecule)
        self.Internals = []
        self.cPrims = []
        self.cVals = []
        self.elem = molecule.elem
        for i in range(molecule.na):
            self.add(CartesianX(i, w=1.0))
            self.add(CartesianY(i, w=1.0))
            self.add(CartesianZ(i, w=1.0))
        if kwargs.get('remove_tr', False):
            raise RuntimeError('Do not use remove_tr with Cartesian coordinates')
        if 'constraints' in kwargs and kwargs['constraints'] is not None:
            raise RuntimeError('Do not use constraints with Cartesian coordinates')

    def guess_hessian(self, xyz):
        return 0.5*np.eye(len(xyz.flatten()))
