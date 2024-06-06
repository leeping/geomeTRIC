"""@package geometric.nifty Nifty functions, originally intended to be imported by any module within ForceBalance.
This file was copied over from ForceBalance to geomeTRIC in order to lighten the dependencies of the latter.

Table of Contents:
- I/O formatting
- Math: Variable manipulation, linear algebra, least squares polynomial fitting
- Pickle: Expand Python's own pickle to accommodate writing XML etree objects
- Commands for submitting things to the Work Queue
- Various file and process management functions
- Development stuff (not commonly used)

Named after the mighty Sniffy Handy Nifty (King Sniffy)

@author Lee-Ping Wang
@date 2018-03-10
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import filecmp
import itertools
import os
import re
import shutil
import sys
from select import select

# distutils has been removed as of python 3.12
if sys.version_info < (3, 8):
    import distutils.dir_util

import numpy as np
from numpy.linalg import multi_dot

# For Python 3 compatibility
try:
    from itertools import zip_longest as zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
import threading
from pickle import Pickler, Unpickler
import tarfile
import time
import subprocess
import math
import six # For six.string_types
from subprocess import PIPE
from collections import OrderedDict, defaultdict

#================================#
#       Set up the logger        #
#================================#
if "forcebalance" in __name__:
    # If this module is part of ForceBalance, use the package level logger
    from .output import *
    package="ForceBalance"
else:
    from logging import *
    # Define two handlers that don't print newline characters at the end of each line
    class RawStreamHandler(StreamHandler):
        """
        Exactly like StreamHandler, except no newline character is printed at the end of each message.
        This is done in order to ensure functions in molecule.py and nifty.py work consistently
        across multiple packages.
        """
        def __init__(self, stream = sys.stdout):
            super(RawStreamHandler, self).__init__(stream)
            self.terminator = ""

    class RawFileHandler(FileHandler):
        """
        Exactly like FileHandler, except no newline character is printed at the end of each message.
        This is done in order to ensure functions in molecule.py and nifty.py work consistently
        across multiple packages.
        """
        def __init__(self, *args, **kwargs):
            super(RawFileHandler, self).__init__(*args, **kwargs)
            self.terminator = ""

    if "geometric" in __name__:
        # This ensures logging behavior is consistent with the rest of geomeTRIC
        logger = getLogger(__name__)
        logger.setLevel(INFO)
        package="geomeTRIC"
    else: # pragma: no cover
        logger = getLogger("NiftyLogger")
        logger.setLevel(INFO)
        handler = RawStreamHandler()
        logger.addHandler(handler)
        if __name__ == "__main__":
            package = "LPW-nifty.py"
        else:
            package = __name__.split('.')[0]

try:
    import bz2
    HaveBZ2 = True
except ImportError:
    logger.warning("bz2 module import failed (used in compressing or decompressing pickle files)\n")
    HaveBZ2 = False

try:
    import gzip
    HaveGZ = True
except ImportError:
    logger.warning("gzip module import failed (used in compressing or decompressing pickle files)\n")
    HaveGZ = False

# The directory that this file lives in
rootdir = os.path.dirname(os.path.abspath(__file__))

# On 2020-05-07, these values were revised to CODATA 2018 values
# hartree-joule relationship   4.359 744 722 2071(85) e-18
# Hartree energy in eV         27.211 386 245 988(53)
# Avogadro constant            6.022 140 76 e23         (exact)
# molar gas constant           8.314 462 618            (exact)
# Boltzmann constant           1.380649e-23             (exact)
# Bohr radius                  5.291 772 109 03(80) e-11
# speed of light in vacuum     299 792 458 (exact)
# reduced Planck's constant    1.054571817e-34 (exact)
# calorie-joule relationship   4.184 J (exact; from NIST)

## Boltzmann constant in kJ mol^-1 k^-1
kb          = 0.008314462618       # Previous value: 0.0083144100163
kb_si       = 1.380649e-23

# Conversion factors
bohr2ang     = 0.529177210903      # Previous value: 0.529177210
ang2bohr     = 1.0 / bohr2ang
au2kcal      = 627.5094740630558   # Previous value: 627.5096080306
kcal2au      = 1.0 / au2kcal
au2kj        = 2625.4996394798254  # Previous value: 2625.5002
kj2au        = 1.0 / au2kj
grad_au2gmx  = 49614.75258920567   # Previous value: 49614.75960959161
grad_gmx2au  = 1.0 / grad_au2gmx
au2ev        = 27.211386245988
ev2au        = 1.0 / au2ev
au2evang     = 51.422067476325886  # Previous value: 51.42209166566339
evang2au     = 1.0 / au2evang
c_lightspeed = 299792458.
hbar         = 1.054571817e-34
avogadro     = 6.02214076e23
au_mass      = 9.1093837015e-31    # Atomic unit of mass in kg
amu_mass     = 1.66053906660e-27   # Atomic mass unit in kg
amu2au       = amu_mass / au_mass
cm2au        = 100 * c_lightspeed * (2*np.pi*hbar) * avogadro / 1000 / au2kj # Multiply to convert cm^-1 to Hartree
ambervel2au  = 9.349961132249932e-04 # Multiply to go from AMBER velocity unit Ang/(1/20.455 ps) to bohr/atu.


## Q-Chem to GMX unit conversion for energy
eqcgmx = au2kj                     # Previous value: 2625.5002
## Q-Chem to GMX unit conversion for force
fqcgmx = -grad_au2gmx              # Previous value: -49621.9


#=========================#
#     I/O formatting      #
#=========================#
# These functions may be useful someday but I have not tested them
# def bzip2(src):
#     dest = src+'.bz2'
#     if not os.path.exists(src):
#         logger.error('File to be compressed does not exist')
#         raise RuntimeError
#     if os.path.exists(dest):
#         logger.error('Archive to be created already exists')
#         raise RuntimeError
#     with open(src, 'rb') as input:
#         with bz2.BZ2File(dest, 'wb', compresslevel=9) as output:
#             copyfileobj(input, output)
#     os.remove(input)

# def bunzip2(src):
#     dest = re.sub('\.bz2$', '', src)
#     if not os.path.exists(src):
#         logger.error('File to be decompressed does not exist')
#         raise RuntimeError
#     if os.path.exists(dest):
#         logger.error('Target path for decompression already exists')
#         raise RuntimeError
#     with bz2.BZ2File(src, 'rb', compresslevel=9) as input:
#         with open(dest, 'wb') as output:
#             copyfileobj(input, output)
#     os.remove(input)

def pvec1d(vec1d, precision=1, format="e", loglevel=INFO):
    """Printout of a 1-D vector.

    @param[in] vec1d a 1-D vector
    """
    v2a = np.array(vec1d)
    for i in range(v2a.shape[0]):
        logger.log(loglevel, "%% .%i%s " % (precision, format) % v2a[i])
    logger.log(loglevel, '\n')

def astr(vec1d, precision=4):
    """ Write an array to a string so we can use it to key a dictionary. """
    return ' '.join([("%% .%ie " % precision % i) for i in vec1d])

def pmat2d(mat2d, precision=1, format="e", loglevel=INFO):
    """Printout of a 2-D array.

    @param[in] mat2d a 2-D array
    """
    m2a = np.array(mat2d)
    for i in range(m2a.shape[0]):
        for j in range(m2a.shape[1]):
            logger.log(loglevel, "%% .%i%s " % (precision, format) % m2a[i][j])
        logger.log(loglevel, '\n')

def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    lzip = [[j for j in i if j is not None] for i in list(zip_longest(*args))]
    return lzip

def encode(l):
    return [[len(list(group)),name] for name, group in itertools.groupby(l)]

def segments(e):
    # Takes encoded input.
    begins = np.array([sum([k[0] for k in e][:j]) for j,i in enumerate(e) if i[1] == 1])
    lens = np.array([i[0] for i in e if i[1] == 1])
    return [(i, i+j) for i, j in zip(begins, lens)]

def commadash(l):
    # Formats a list like [27, 28, 29, 30, 31, 88, 89, 90, 91, 100, 136, 137, 138, 139]
    # into '27-31,88-91,100,136-139
    L = sorted(l)
    if len(L) == 0:
        return "(empty)"
    L.append(L[-1]+1)
    LL = [i in L for i in range(L[-1])]
    return ','.join('%i-%i' % (i[0]+1,i[1]) if (i[1]-1 > i[0]) else '%i' % (i[0]+1) for i in segments(encode(LL)))

def uncommadash(s):
    # Takes a string like '27-31,88-91,100,136-139'
    # and turns it into a list like [26, 27, 28, 29, 30, 87, 88, 89, 90, 99, 135, 136, 137, 138]
    L = []
    try:
        for w in s.split(','):
            ws = w.split('-')
            a = int(ws[0])-1
            if len(ws) == 1:
                b = int(ws[0])
            elif len(ws) == 2:
                b = int(ws[1])
            else:
                logger.warning("Dash-separated list cannot exceed length 2\n")
                raise RuntimeError
            if a < 0 or b <= 0 or b <= a:
                if a < 0 or b <= 0:
                    raise RuntimeError("Items in list cannot be zero or negative: %d %d\n" % (a, b))
                else:
                    raise RuntimeError("Second number cannot be smaller than first: %d %d\n" % (a, b))
            newL = range(a,b)
            if any([i in L for i in newL]):
                raise RuntimeError("Duplicate entries found in list\n")
            L += newL
        if sorted(L) != L:
            raise RuntimeError("List is out of order\n")
    except:
        raise RuntimeError('Invalid string for converting to list of numbers: %s\n' % s)
    return L

def natural_sort(l):
    """ Return a natural sorted list. """
    # Convert a character to a digit or a lowercase character
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # Split string into "integer" and "noninteger" fields and convert each one
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    # Sort strings using these keys in descending order of importance, I guess.
    return sorted(l, key = alphanum_key)

def printcool(text,sym="#",bold=False,color=2,ansi=None,bottom='-',minwidth=50,center=True,sym2="="):
    """Cool-looking printout for slick formatting of output.

    @param[in] text The string that the printout is based upon.  This function
    will print out the string, ANSI-colored and enclosed in the symbol
    for example:\n
    <tt> ################# </tt>\n
    <tt> ### I am cool ### </tt>\n
    <tt> ################# </tt>
    @param[in] sym The surrounding symbol\n
    @param[in] bold Whether to use bold print

    @param[in] color The ANSI color:\n
    1 red\n
    2 green\n
    3 yellow\n
    4 blue\n
    5 magenta\n
    6 cyan\n
    7 white

    @param[in] bottom The symbol for the bottom bar

    @param[in] minwidth The minimum width for the box, if the text is very short
    then we insert the appropriate number of padding spaces

    @return bar The bottom bar is returned for the user to print later, e.g. to mark off a 'section'
    """
    def newlen(l):
        return len(re.sub(r"\x1b\[[0-9;]*m","",l))
    text = text.split('\n')
    width = max(minwidth,max([newlen(line) for line in text]))
    bar = ''.join([sym2 for i in range(width + 6)])
    bar = sym + bar + sym
    #bar = ''.join([sym for i in range(width + 8)])
    logger.info('\r'+bar + '\n')
    for ln, line in enumerate(text):
        if type(center) is list: c1 = center[ln]
        else: c1 = center
        if c1:
            padleft = ' ' * (int((width - newlen(line))/2))
        else:
            padleft = ''
        padright = ' '* (width - newlen(line) - len(padleft))
        if ansi is not None:
            ansi = str(ansi)
            logger.info("%s| \x1b[%sm%s " % (sym, ansi, padleft)+line+" %s\x1b[0m |%s\n" % (padright, sym))
        elif color is not None:
            if color == 0 and bold:
                logger.info("%s| \x1b[1m%s " % (sym, padleft) + line + " %s\x1b[0m |%s\n" % (padright, sym))
            elif color == 0:
                logger.info("%s| %s " % (sym, padleft)+line+" %s |%s\n" % (padright, sym))
            else:
                logger.info("%s| \x1b[%s9%im%s " % (sym, bold and "1;" or "", color, padleft)+line+" %s\x1b[0m |%s\n" % (padright, sym))
            # if color == 3 or color == 7:
            #     print "%s\x1b[40m\x1b[%s9%im%s" % (''.join([sym for i in range(3)]), bold and "1;" or "", color, padleft),line,"%s\x1b[0m%s" % (padright, ''.join([sym for i in range(3)]))
            # else:
            #     print "%s\x1b[%s9%im%s" % (''.join([sym for i in range(3)]), bold and "1;" or "", color, padleft),line,"%s\x1b[0m%s" % (padright, ''.join([sym for i in range(3)]))
        else:
            warn_press_key("Inappropriate use of printcool")
    logger.info(bar + '\n')
    botbar = ''.join([bottom for i in range(width + 8)])
    return botbar + '\n'

def printcool_dictionary(Dict,title="Dictionary Keys : Values",bold=False,color=2,keywidth=25,topwidth=50,center=True,leftpad=0):
    """See documentation for printcool; this is a nice way to print out keys/values in a dictionary.

    The keys in the dictionary are sorted before printing out.

    @param[in] dict The dictionary to be printed
    @param[in] title The title of the printout
    """
    if Dict is None: return
    bar = printcool(title,bold=bold,color=color,minwidth=topwidth,center=center)
    def magic_string(str):
        # This cryptic command returns a string with the number of characters specified as a variable. :P
        # Useful for printing nice-looking dictionaries, i guess.
        # print "\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"'))
        return eval("\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"')))
    if isinstance(Dict, OrderedDict):
        logger.info('\n'.join([' '*leftpad + "%s %s " % (magic_string(str(key)),str(Dict[key])) for key in Dict if Dict[key] is not None]))
    else:
        logger.info('\n'.join([' '*leftpad + "%s %s " % (magic_string(str(key)),str(Dict[key])) for key in sorted([i for i in Dict]) if Dict[key] is not None]))
    logger.info("\n%s" % bar)

#===============================#
#| Math: Variable manipulation |#
#===============================#
def isint(word):
    """ONLY matches integers! If you have a decimal point? None shall pass!

    @param[in] word String (for instance, '123', '153.0', '2.', '-354')
    @return answer Boolean which specifies whether the string is an integer (only +/- sign followed by digits)

    """
    try:
        word = str(word)
    except:
        return False
    return re.match('^[-+]?[0-9]+$', word)

def isfloat(word):
    """Matches ANY number; it can be a decimal, scientific notation, what have you
    CAUTION - this will also match an integer.

    @param[in] word String (for instance, '123', '153.0', '2.', '-354')
    @return answer Boolean which specifies whether the string is any number

    """
    try: word = str(word)
    except: return False
    if len(word) == 0: return False
    return re.match(r'^[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?$',word)

def isdecimal(word):
    """Matches things with a decimal only; see isint and isfloat.

    @param[in] word String (for instance, '123', '153.0', '2.', '-354')
    @return answer Boolean which specifies whether the string is a number with a decimal point

    """
    try: word = str(word)
    except: return False
    return isfloat(word) and not isint(word)

def floatornan(word):
    """Returns a big number if we encounter NaN or inf.

    @param[in] word The string to be converted
    @return answer The string converted to a float; if not a float, return 1e10
    @todo I could use suggestions for making this better.
    """
    big = 1e100
    if isfloat(word):
        return float(word)
    elif word.lower() in ['inf', 'nan']:
        logger.info("Setting %s to % .1e\n" % (word, big))
        return big
    else:
        raise ValueError("%s cannot be converted to a number" % word)

def col(vec):
    """
    Given any list, array, or matrix, return a 1-column 2D array.

    Input:
    vec  = The input vector that is to be made into a column

    Output:
    A 1-column 2D array
    """
    return np.array(vec).reshape(-1, 1)

def row(vec):
    """Given any list, array, or matrix, return a 1-row 2D array.

    @param[in] vec The input vector that is to be made into a row

    @return answer A 1-row 2D array
    """
    return np.array(vec).reshape(1, -1)

def flat(vec):
    """Given any list, array, or matrix, return a single-index array.

    @param[in] vec The data to be flattened
    @return answer The flattened data
    """
    return np.array(vec).reshape(-1)

def est124(val):
    """Given any positive floating point value, return a value [124]e+xx
    that is closest to it in the log space.
    """
    log = np.log10(val)
    logint = math.floor(log)
    logfrac = log - logint
    log1 = 0.0
    log2 = 0.3010299956639812
    log4 = 0.6020599913279624
    log10 = 1.0
    if logfrac < 0.5*(log1+log2):
        fac = 1.0
    elif logfrac < 0.5*(log2+log4):
        fac = 2.0
    elif logfrac < 0.5*(log4+log10):
        fac = 4.0
    else:
        fac = 10.0
    return fac*10**logint

def monotonic_decreasing(arr, start=None, end=None, verbose=False):
    """
    Return the indices of an array corresponding to strictly monotonic
    decreasing behavior.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    start : int
        Starting index, default 0
    end : int
        Ending index, included in loop, default len(arr) - 1

    Returns
    -------
    indices : numpy.ndarray
        Selected indices
    """
    if start is None:
        start = 0
    if end is None:
        end = len(arr) - 1
    a0 = arr[start]
    idx = [start]
    if verbose: logger.info("Starting @ %i : %.6f\n" % (start, arr[start]))
    if end > start:
        i = start+1
        while i <= end:
            if arr[i] < a0:
                a0 = arr[i]
                idx.append(i)
                if verbose: logger.info("Including  %i : %.6f\n" % (i, arr[i]))
            else:
                if verbose: logger.info("Excluding  %i : %.6f\n" % (i, arr[i]))
            i += 1
    if end < start:
        i = start-1
        while i >= end:
            if arr[i] < a0:
                a0 = arr[i]
                idx.append(i)
                if verbose: logger.info("Including  %i : %.6f\n" % (i, arr[i]))
            else:
                if verbose: logger.info("Excluding  %i : %.6f\n" % (i, arr[i]))
            i -= 1
    return np.array(idx)

#====================================#
#| Math: Vectors and linear algebra |#
#====================================#
def orthogonalize(vec1, vec2):
    """Given two vectors vec1 and vec2, project out the component of vec1
    that is along the vec2-direction.

    @param[in] vec1 The projectee (i.e. output is some modified version of vec1)
    @param[in] vec2 The projector (component subtracted out from vec1 is parallel to this)
    @return answer A copy of vec1 but with the vec2-component projected out.
    """
    v2u = vec2/np.linalg.norm(vec2)
    return vec1 - v2u*np.dot(vec1, v2u)

def invert_svd(X,thresh=1e-12):

    """

    Invert a matrix using singular value decomposition.
    @param[in] X The 2-D NumPy array containing the matrix to be inverted
    @param[in] thresh The SVD threshold; eigenvalues below this are not inverted but set to zero
    @return Xt The 2-D NumPy array containing the inverted matrix

    """

    u,s,vh = np.linalg.svd(X, full_matrices=0)
    uh     = np.transpose(u)
    v      = np.transpose(vh)
    si     = s.copy()
    for i in range(s.shape[0]):
        # print("SVD : %i -> %.3e" % (i, s[i]))
        if abs(s[i]) > thresh:
            si[i] = 1./s[i]
        else:
            si[i] = 0.0
    si     = np.diag(si)
    Xt     = multi_dot([v, si, uh])
    return Xt

#==============================#
#|    Linear least squares    |#
#==============================#
def get_least_squares(x, y, w = None, thresh=1e-12):
    """
    @code
     __                  __
    |                      |
    | 1 (x0) (x0)^2 (x0)^3 |
    | 1 (x1) (x1)^2 (x1)^3 |
    | 1 (x2) (x2)^2 (x2)^3 |
    | 1 (x3) (x3)^2 (x3)^3 |
    | 1 (x4) (x4)^2 (x4)^3 |
    |__                  __|

    @endcode

    @param[in] X (2-D array) An array of X-values (see above)
    @param[in] Y (array) An array of Y-values (only used in getting the least squares coefficients)
    @param[in] w (array) An array of weights, hopefully normalized to one.
    @param[out] Beta The least-squares coefficients
    @param[out] Hat The hat matrix that takes linear combinations of data y-values to give fitted y-values (weights)
    @param[out] yfit The fitted y-values
    @param[out] MPPI The Moore-Penrose pseudoinverse (multiply by Y to get least-squares coefficients, multiply by dY/dk to get derivatives of least-squares coefficients)
    """
    # X is a 'tall' matrix.
    X = np.array(x)
    if len(X.shape) == 1:
        X = X[:,np.newaxis]
    Y = col(y)
    n_x = X.shape[0]
    n_fit = X.shape[1]
    if n_fit > n_x:
        logger.warning("Argh? It seems like this problem is underdetermined!\n")
    # Build the weight matrix.
    if w is not None:
        if len(w) != n_x:
            warn_press_key("The weight array length (%i) must be the same as the number of 'X' data points (%i)!" % len(w), n_x)
        w /= np.mean(w)
        WH = np.diag(w**0.5)
    else:
        WH = np.eye(n_x)
    # Make the Moore-Penrose Pseudoinverse.
    # if n_fit == n_x:
    #     MPPI = np.linalg.inv(WH*X)
    # else:
    # This resembles the formula (X'WX)^-1 X' W^1/2
    MPPI = np.linalg.pinv(np.dot(WH, X))
    Beta = multi_dot([MPPI, WH, Y])
    Hat = multi_dot([WH, X, MPPI])
    yfit = flat(np.dot(Hat, Y))
    # Return three things: the least-squares coefficients, the hat matrix (turns y into yfit), and yfit
    # We could get these all from MPPI, but I might get confused later on, so might as well do it here :P
    return np.array(Beta).flatten(), np.array(Hat), np.array(yfit).flatten(), np.array(MPPI)

#===========================================#
#| John's statisticalInefficiency function |#
#===========================================#
def statisticalInefficiency(A_n, B_n=None, fast=False, mintime=3, warn=True):

    """
    Compute the (cross) statistical inefficiency of (two) timeseries.

    Notes
      The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
      The fast method described in Ref [1] is used to compute g.

    References
      [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
      histogram analysis method for the analysis of simulated and parallel tempering simulations.
      JCTC 3(1):26-41, 2007.

    Examples

    Compute statistical inefficiency of timeseries data with known correlation time.

    >>> import timeseries
    >>> A_n = timeseries.generateCorrelatedTimeseries(N=100000, tau=5.0)
    >>> g = statisticalInefficiency(A_n, fast=True)

    @param[in] A_n (required, numpy array) - A_n[n] is nth value of
    timeseries A.  Length is deduced from vector.

    @param[in] B_n (optional, numpy array) - B_n[n] is nth value of
    timeseries B.  Length is deduced from vector.  If supplied, the
    cross-correlation of timeseries A and B will be estimated instead of
    the autocorrelation of timeseries A.

    @param[in] fast (optional, boolean) - if True, will use faster (but
    less accurate) method to estimate correlation time, described in
    Ref. [1] (default: False)

    @param[in] mintime (optional, int) - minimum amount of correlation
    function to compute (default: 3) The algorithm terminates after
    computing the correlation time out to mintime when the correlation
    function furst goes negative.  Note that this time may need to be
    increased if there is a strong initial negative peak in the
    correlation function.

    @return g The estimated statistical inefficiency (equal to 1 + 2
    tau, where tau is the correlation time).  We enforce g >= 1.0.

    """
    # Create numpy copies of input arguments.
    A_n = np.array(A_n)
    if B_n is not None:
        B_n = np.array(B_n)
    else:
        B_n = np.array(A_n)
    # Get the length of the timeseries.
    N = A_n.shape[0]
    # Be sure A_n and B_n have the same dimensions.
    if A_n.shape != B_n.shape:
        logger.error('A_n and B_n must have same dimensions.\n')
        raise ParameterError
    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0
    # Compute mean of each timeseries.
    mu_A = A_n.mean()
    mu_B = B_n.mean()
    # Make temporary copies of fluctuation from mean.
    dA_n = A_n.astype(np.float64) - mu_A
    dB_n = B_n.astype(np.float64) - mu_B
    # Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
    sigma2_AB = (dA_n * dB_n).mean() # standard estimator to ensure C(0) = 1
    # Trap the case where this covariance is zero, and we cannot proceed.
    if sigma2_AB == 0:
        if warn:
            logger.warning('Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency\n')
        return 1.0
    # Accumulate the integrated correlation time by computing the normalized correlation time at
    # increasing values of t.  Stop accumulating if the correlation function goes negative, since
    # this is unlikely to occur unless the correlation function has decayed to the point where it
    # is dominated by noise and indistinguishable from zero.
    t = 1
    increment = 1
    while t < N-1:
        # compute normalized fluctuation correlation function at time t
        C = sum( dA_n[0:(N-t)]*dB_n[t:N] + dB_n[0:(N-t)]*dA_n[t:N] ) / (2.0 * float(N-t) * sigma2_AB)
        # Terminate if the correlation function has crossed zero and we've computed the correlation
        # function at least out to 'mintime'.
        if (C <= 0.0) and (t > mintime):
            break
        # Accumulate contribution to the statistical inefficiency.
        g += 2.0 * C * (1.0 - float(t)/float(N)) * float(increment)
        # Increment t and the amount by which we increment t.
        t += increment
        # Increase the interval if "fast mode" is on.
        if fast: increment += 1
    # g must be at least unity
    if g < 1.0: g = 1.0
    # Return the computed statistical inefficiency.
    return g

def mean_stderr(ts):
    """Return mean and standard deviation of a time series ts."""
    return np.mean(ts), \
      np.std(ts)*np.sqrt(statisticalInefficiency(ts, warn=False)/len(ts))

# Slices a 2D array of data by column.  The new array is fed into the statisticalInefficiency function.
def multiD_statisticalInefficiency(A_n, B_n=None, fast=False, mintime=3, warn=True):
    n_row = A_n.shape[0]
    n_col = A_n.shape[-1]
    multiD_sI = np.zeros((n_row, n_col))
    for col in range(n_col):
        if B_n is None:
            multiD_sI[:,col] = statisticalInefficiency(A_n[:,col], B_n, fast, mintime, warn)
        else:
            multiD_sI[:,col] = statisticalInefficiency(A_n[:,col], B_n[:,col], fast, mintime, warn)
    return multiD_sI

#========================================#
#|      Loading compressed pickles      |#
#========================================#

def lp_dump(obj, fnm, protocol=0):
    """ Write an object to a zipped pickle file specified by the path. """
    # Safeguard against overwriting files?  Nah.
    # if os.path.exists(fnm):
    #     logger.error("lp_dump cannot write to an existing path")
    #     raise IOError
    if os.path.islink(fnm):
        logger.warning("Trying to write to a symbolic link %s, removing it first\n" % fnm)
        os.unlink(fnm)
    if HaveGZ:
        f = gzip.GzipFile(fnm, 'wb')
    elif HaveBZ2:
        f = bz2.BZ2File(fnm, 'wb')
    else:
        f = open(fnm, 'wb')
    Pickler(f, protocol).dump(obj)
    f.close()

def lp_load(fnm):
    """ Read an object from a bzipped file specified by the path. """
    if not os.path.exists(fnm):
        logger.error("lp_load cannot read from a path that doesn't exist (%s)" % fnm)
        raise IOError

    def load_uncompress():
        logger.warning("Compressed file loader failed, attempting to read as uncompressed file\n")
        f = open(fnm, 'rb')
        try:
            answer = Unpickler(f).load()
        except UnicodeDecodeError:
            answer = Unpickler(f, encoding='latin1').load()
        f.close()
        return answer

    def load_bz2():
        f = bz2.BZ2File(fnm, 'rb')
        try:
            answer = Unpickler(f).load()
        except UnicodeDecodeError:
            answer = Unpickler(f, encoding='latin1').load()
        f.close()
        return answer

    def load_gz():
        f = gzip.GzipFile(fnm, 'rb')
        try:
            answer = Unpickler(f).load()
        except UnicodeDecodeError:
            answer = Unpickler(f, encoding='latin1').load()
        f.close()
        return answer

    if HaveGZ:
        try:
            answer = load_gz()
        except:
            if HaveBZ2:
                try:
                    answer = load_bz2()
                except:
                    answer = load_uncompress()
            else:
                answer = load_uncompress()
    elif HaveBZ2:
        try:
            answer = load_bz2()
        except:
            answer = load_uncompress()
    else:
        answer = load_uncompress()
    return answer

#==============================#
#|      Work Queue stuff      |#
#==============================#
try:
    try:
        import ndcctools.work_queue as work_queue
    except ImportError:
        import work_queue
except:
    pass
    #logger.warning("Work Queue library import fail (You can't queue up jobs using Work Queue)\n")

# Global variable corresponding to the Work Queue object
WORK_QUEUE = None

# Global variable containing a mapping from target names to Work Queue task IDs
WQIDS = defaultdict(list)

def getWorkQueue():
    global WORK_QUEUE
    return WORK_QUEUE

def getWQIds():
    global WQIDS
    return WQIDS

def createWorkQueue(wq_port, debug=True, name=package):
    global WORK_QUEUE
    if debug:
        work_queue.set_debug_flag('all')
    WORK_QUEUE = work_queue.WorkQueue(port=wq_port)
    WORK_QUEUE.specify_name(name)
    # QYD: prefer the worker that is fastest in previous tasks
    # another choice is first-come-first serve: WORK_QUEUE_SCHEDULE_FCFS
    WORK_QUEUE.specify_algorithm(work_queue.WORK_QUEUE_SCHEDULE_TIME)
    # QYD: We don't want to specify the following extremely long keepalive times
    # because they will prevent checking "dead" workers, causing the program to wait forever
    #WORK_QUEUE.specify_keepalive_timeout(8640000)
    #WORK_QUEUE.specify_keepalive_interval(8640000)

def destroyWorkQueue():
    # Convenience function to destroy the Work Queue objects.
    global WORK_QUEUE, WQIDS
    if hasattr(WORK_QUEUE, '_free'): 
        print("Freeing Work Queue")
        WORK_QUEUE._free()
    WORK_QUEUE = None
    WQIDS = defaultdict(list)

def queue_up(wq, command, input_files, output_files, tag=None, tgt=None, verbose=True, print_time=60):
    """
    Submit a job to the Work Queue.

    @param[in] wq (Work Queue Object)
    @param[in] command (string) The command to run on the remote worker.
    @param[in] input_files (list of files) A list of locations of the input files.
    @param[in] output_files (list of files) A list of locations of the output files.
    """
    global WQIDS
    task = work_queue.Task(command)
    cwd = os.getcwd()
    for f in input_files:
        lf = os.path.join(cwd,f)
        task.specify_input_file(lf,f,cache=False)
    for f in output_files:
        lf = os.path.join(cwd,f)
        task.specify_output_file(lf,f,cache=False)
    if tag is None: tag = command
    task.specify_tag(tag)
    task.print_time = print_time
    taskid = wq.submit(task)
    if verbose:
        logger.info("Submitting command '%s' to the Work Queue, %staskid %i\n" % (command, "tag %s, " % tag if tag != command else "", taskid))
    if tgt is not None:
        WQIDS[tgt.name].append(taskid)
    else:
        WQIDS["None"].append(taskid)

def queue_up_src_dest(wq, command, input_files, output_files, tag=None, tgt=None, verbose=True, print_time=60):
    """
    Submit a job to the Work Queue.  This function is a bit fancier in that we can explicitly
    specify where the input files come from, and where the output files go to.

    @param[in] wq (Work Queue Object)
    @param[in] command (string) The command to run on the remote worker.
    @param[in] input_files (list of 2-tuples) A list of local and
    remote locations of the input files.
    @param[in] output_files (list of 2-tuples) A list of local and
    remote locations of the output files.
    """
    global WQIDS
    task = work_queue.Task(command)
    for f in input_files:
        # print f[0], f[1]
        task.specify_input_file(f[0],f[1],cache=False)
    for f in output_files:
        # print f[0], f[1]
        task.specify_output_file(f[0],f[1],cache=False)
    if tag is None: tag = command
    task.specify_tag(tag)
    task.print_time = print_time
    taskid = wq.submit(task)
    if verbose:
        logger.info("Submitting command '%s' to the Work Queue, taskid %i\n" % (command, taskid))
    if tgt is not None:
        WQIDS[tgt.name].append(taskid)
    else:
        WQIDS["None"].append(taskid)

def wq_wait1(wq, wait_time=10, wait_intvl=1, print_time=60, verbose=False):
    """ This function waits ten seconds to see if a task in the Work Queue has finished. """
    global WQIDS
    if verbose: logger.info('---\n')
    if wait_intvl >= wait_time:
        wait_time = wait_intvl
        numwaits = 1
    else:
        numwaits = int(wait_time/wait_intvl)
    for sec in range(numwaits):
        task = wq.wait(wait_intvl)
        if task:
            exectime = task.cmd_execution_time/1000000
            if verbose:
                logger.info('A job has finished!\n')
                logger.info('Job name = ' + task.tag + '\n')
                logger.info('command = ' + task.command + '\n')
                logger.info("return_status = " + str(task.return_status)) # the process's return status
                logger.info("result = " + str(task.result)) # nonzero if the specified output file doesn't exist
                logger.info("host = " + task.hostname + '\n')
                logger.info("execution time = %.3f" % exectime)
                logger.info("total_bytes_transferred = %i" % task.total_bytes_transferred + '\n')
            if task.result != 0: # pragma: no cover
                oldid = task.id
                oldhost = task.hostname
                tgtname = "None"
                for tnm in WQIDS:
                    if task.id in WQIDS[tnm]:
                        tgtname = tnm
                        WQIDS[tnm].remove(task.id)
                taskid = wq.submit(task)
                logger.warning("Task '%s' (task %i) failed on host %s (%i seconds), resubmitted: taskid %i\n" % (task.tag, oldid, oldhost, exectime, taskid))
                WQIDS[tgtname].append(taskid)
            else:
                if hasattr(task, 'print_time'):
                    print_time = task.print_time
                if exectime > print_time: # Assume that we're only interested in printing jobs that last longer than a minute.
                    logger.info("Task '%s' (task %i) finished successfully on host %s (%i seconds)\n" % (task.tag, task.id, task.hostname, exectime))
                for tnm in WQIDS:
                    if task.id in WQIDS[tnm]:
                        WQIDS[tnm].remove(task.id)
                del task

        # LPW 2018-09-10 Updated to use stats fields from CCTools 6.2.10
        # Please upgrade CCTools version if errors are encountered during runtime.
        if verbose:
            logger.info("Workers: %i init, %i idle, %i busy, %i total joined, %i total removed\n" \
                % (wq.stats.workers_init, wq.stats.workers_idle, wq.stats.workers_busy, wq.stats.workers_joined, wq.stats.workers_removed))
            logger.info("Tasks: %i running, %i waiting, %i dispatched, %i submitted, %i total complete\n" \
                % (wq.stats.tasks_running, wq.stats.tasks_waiting, wq.stats.tasks_dispatched, wq.stats.tasks_submitted, wq.stats.tasks_done))
            logger.info("Data: %i / %i kb sent/received\n" % (int(wq.stats.bytes_sent/1024), int(wq.stats.bytes_received/1024)))
        else:
            logger.info("\r%s : %i/%i workers busy; %i/%i jobs complete  \r" %\
            (time.ctime(), wq.stats.workers_busy, wq.stats.workers_connected, wq.stats.tasks_done, wq.stats.tasks_submitted))
            if time.time() - wq_wait1.t0 > 900:
                wq_wait1.t0 = time.time()
                logger.info('\n')
wq_wait1.t0 = time.time()

def wq_wait(wq, wait_time=10, wait_intvl=10, print_time=60, verbose=False):
    """ This function waits until the work queue is completely empty. """
    while not wq.empty():
        wq_wait1(wq, wait_time=wait_time, wait_intvl=wait_intvl, print_time=print_time, verbose=verbose)

#=====================================#
#| File and process management stuff |#
#=====================================#
def click():
    """ Stopwatch function for timing. """
    ans = time.time() - click.t0
    click.t0 = time.time()
    return ans
click.t0 = time.time()

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

# Back up a file.
def bak(path, dest=None, cwd=None, basename=None, start=1):
    oldf = path
    newf = None
    if cwd != None:
        if not os.path.exists(cwd):
            raise RuntimeError("%s is not an existing folder" % cwd)
        old_d = os.getcwd()
        os.chdir(cwd)
    if os.path.exists(path):
        dnm, fnm = os.path.split(path)
        if dnm == '' : dnm = '.'
        base, ext = os.path.splitext(fnm)
        if basename: base = basename
        if dest is None:
            dest = dnm
        if not os.path.isdir(dest): os.makedirs(dest)
        i = start
        while True:
            fnm = "%s_%i%s" % (base,i,ext)
            newf = os.path.join(dest, fnm)
            if not os.path.exists(newf): break
            i += 1
        logger.info("Backing up %s -> %s\n" % (oldf, newf))
        shutil.move(oldf,newf)
    if cwd != None:
        os.chdir(old_d)
    return newf

# Purpose: Given a file name and/or an extension, do one of the following:
# 1) If provided a file name, check the file, crash if not exist and err==True.  Return the file name.
# 2) If list is empty but extension is provided, check if one file exists that matches
# the extension.  If so, return the file name.
# 3) If list is still empty and err==True, then crash with an error.
def onefile(fnm=None, ext=None, err=False):
    if fnm is None and ext is None:
        if err:
            logger.error("Must provide either filename or extension to onefile()")
            raise RuntimeError
        else:
            return None
    if fnm is not None:
        if os.path.exists(fnm):
            if os.path.dirname(os.path.abspath(fnm)) != os.getcwd():
                fsrc = os.path.abspath(fnm)
                fdest = os.path.join(os.getcwd(), os.path.basename(fnm))
                #-----
                # If the file path doesn't correspond to the current directory, copy the file over
                # If the file exists in the current directory already and it's different, then crash.
                #-----
                if os.path.exists(fdest):
                    if not filecmp.cmp(fsrc, fdest):
                        logger.error("onefile() will not overwrite %s with %s\n" % (os.path.join(os.getcwd(), os.path.basename(fnm)),os.path.abspath(fnm)))
                        raise RuntimeError
                    else:
                        logger.info("\x1b[93monefile() says the files %s and %s are identical\x1b[0m\n" % (os.path.abspath(fnm), os.getcwd()))
                else:
                    logger.info("\x1b[93monefile() will copy %s to %s\x1b[0m\n" % (os.path.abspath(fnm), os.getcwd()))
                    shutil.copy2(fsrc, fdest)
            return os.path.basename(fnm)
        elif err==True or ext is None:
            logger.error("File specified by %s does not exist!" % fnm)
            raise RuntimeError
        elif ext is not None:
            warn_once("File specified by %s does not exist - will try to autodetect .%s extension" % (fnm, ext))
    answer = None
    cwd = os.getcwd()
    ls = [i for i in os.listdir(cwd) if i.endswith('.%s' % ext)]
    if len(ls) != 1:
        if err:
            logger.error("Cannot find a unique file with extension .%s in %s (%i found; %s)" % (ext, cwd, len(ls), ' '.join(ls)))
            raise RuntimeError
        else:
            warn_once("Cannot find a unique file with extension .%s in %s (%i found; %s)" %
                      (ext, cwd, len(ls), ' '.join(ls)), warnhash = "Found %i .%s files" % (len(ls), ext))
    else:
        answer = os.path.basename(ls[0])
        warn_once("Autodetected %s in %s" % (answer, cwd), warnhash = "Autodetected %s" % answer)
    return answer

# Purpose: Given a file name / file list and/or an extension, do one of the following:
# 1) If provided a file list, check each file in the list
# and crash if any file does not exist.  Return the list.
# 2) If provided a file name, check the file and crash if the file
# does not exist.  Return a length-one list with the file name.
# 3) If list is empty but extension is provided, check for files that
# match the extension.  If so, append them to the list.
# 4) If list is still empty and err==True, then crash with an error.
def listfiles(fnms=None, ext=None, err=False, dnm=None):
    answer = []
    cwd = os.path.abspath(os.getcwd())
    if dnm is not None:
        os.chdir(dnm)
    if isinstance(fnms, list):
        for i in fnms:
            if not os.path.exists(i):
                logger.error('Specified %s but it does not exist' % i)
                os.chdir(cwd)
                raise RuntimeError
            answer.append(i)
    elif isinstance(fnms, six.string_types):
        if not os.path.exists(fnms):
            logger.error('Specified %s but it does not exist' % fnms)
            os.chdir(cwd)
            raise RuntimeError
        answer = [fnms]
    elif fnms is not None:
        logger.info(str(fnms))
        logger.error('First argument to listfiles must be a list, a string, or None')
        os.chdir(cwd)
        raise RuntimeError
    if answer == [] and ext is not None:
        answer = [os.path.basename(i) for i in os.listdir(os.getcwd()) if i.endswith('.%s' % ext)]
    if answer == [] and err:
        logger.error('listfiles function failed to come up with a file! (fnms = %s ext = %s)' % (str(fnms), str(ext)))
        os.chdir(cwd)
        raise RuntimeError

    for ifnm, fnm in enumerate(answer):
        if os.path.dirname(os.path.abspath(fnm)) != os.getcwd():
            fsrc = os.path.abspath(fnm)
            fdest = os.path.join(os.getcwd(), os.path.basename(fnm))
            logger.info("fsrc %s fdest %s\n" % (fsrc, fdest))
            #-----
            # If the file path doesn't correspond to the current directory, copy the file over
            # If the file exists in the current directory already and it's different, then crash.
            #-----
            if os.path.exists(fdest):
                if not filecmp.cmp(fsrc, fdest):
                    logger.error("onefile() will not overwrite %s with %s\n" % (os.path.join(os.getcwd(), os.path.basename(fnm)),os.path.abspath(fnm)))
                    os.chdir(cwd)
                    raise RuntimeError
                else:
                    logger.info("\x1b[93monefile() says the files %s and %s are identical\x1b[0m\n" % (os.path.abspath(fnm), os.getcwd()))
                    answer[ifnm] = os.path.basename(fnm)
            else:
                logger.info("\x1b[93monefile() will copy %s to %s\x1b[0m\n" % (os.path.abspath(fnm), os.getcwd()))
                shutil.copy2(fsrc, fdest)
                answer[ifnm] = os.path.basename(fnm)
    os.chdir(cwd)
    return answer

def extract_tar(tarfnm, fnms, force=False):
    """
    Extract a list of files from .tar archive with any compression.
    The file is extracted to the base folder of the archive.

    Parameters
    ----------
    tarfnm :
        Name of the archive file.
    fnms : str or list
        File names to be extracted.
    force : bool, optional
        If true, then force extraction of file even if they already exist on disk.
    """
    # Get path of tar file.
    fdir = os.path.abspath(os.path.dirname(tarfnm))
    # If all files exist, then return - no need to extract.
    if (not force) and all([os.path.exists(os.path.join(fdir, f)) for f in fnms]): return
    # If the tar file doesn't exist or isn't valid, do nothing.
    if not os.path.exists(tarfnm): return
    if not tarfile.is_tarfile(tarfnm): return
    # Check type of fnms argument.
    if isinstance(fnms, six.string_types): fnms = [fnms]
    # Load the tar file.
    arch = tarfile.open(tarfnm, 'r')
    # Extract only the files we have (to avoid an exception).
    all_members = arch.getmembers()
    all_names = [f.name for f in all_members]
    members = [f for f in all_members if f.name in fnms]
    # Extract files to the destination.
    arch.extractall(fdir, members=members)

def GoInto(Dir):
    if os.path.exists(Dir):
        if os.path.isdir(Dir): pass
        else:
            logger.error("Tried to create directory %s, it exists but isn't a directory\n" % newdir)
            raise RuntimeError
    else:
        os.makedirs(Dir)
    os.chdir(Dir)

def allsplit(Dir):
    # Split a directory into all directories involved.
    s = os.path.split(os.path.normpath(Dir))
    if s[1] == '' or s[1] == '.' : return []
    return allsplit(s[0]) + [s[1]]

def Leave(Dir):
    if os.path.split(os.getcwd())[1] != Dir:
        logger.error("Trying to leave directory %s, but we're actually in directory %s (check your code)\n" % (Dir,os.path.split(os.getcwd())[1]))
        raise RuntimeError
    for i in range(len(allsplit(Dir))):
        os.chdir('..')

# Dictionary containing specific error messages for specific missing files or file patterns
specific_lst = [(['mdrun','grompp','trjconv','g_energy','g_traj'], "Make sure to install GROMACS and add it to your path (or set the gmxpath option)"),
                (['force.mdin', 'stage.leap'], "This file is needed for setting up AMBER force matching targets"),
                (['conf.pdb', 'mono.pdb'], "This file is needed for setting up OpenMM condensed phase property targets"),
                (['liquid.xyz', 'liquid.key', 'mono.xyz', 'mono.key'], "This file is needed for setting up OpenMM condensed phase property targets"),
                (['dynamic', 'analyze', 'minimize', 'testgrad', 'vibrate', 'optimize', 'polarize', 'superpose'], "Make sure to install TINKER and add it to your path (or set the tinkerpath option)"),
                (['runcuda.sh', 'npt.py', 'npt_tinker.py'], "This file belongs in the ForceBalance source directory, not sure why it is missing"),
                (['input.xyz'], "This file is needed for TINKER molecular property targets"),
                (['.*key$', '.*xyz$'], "I am guessing this file is probably needed by TINKER"),
                (['.*gro$', '.*top$', '.*itp$', '.*mdp$', '.*ndx$'], "I am guessing this file is probably needed by GROMACS")
                ]

# Build a dictionary mapping all of the keys in the above lists to their error messages
specific_dct = dict(list(itertools.chain(*[[(j,i[1]) for j in i[0]] for i in specific_lst])))

def MissingFileInspection(fnm):
    fnm = os.path.split(fnm)[1]
    answer = ""
    for key in specific_dct:
        if answer == "":
            answer += "\n"
        if re.match(key, fnm):
            answer += "%s\n" % specific_dct[key]
    return answer

def wopen(dest, binary=False):
    """ If trying to write to a symbolic link, remove it first. """
    if os.path.islink(dest):
        logger.warning("Trying to write to a symbolic link %s, removing it first\n" % dest)
        os.unlink(dest)
    if binary:
        return open(dest,'wb')
    else:
        return open(dest,'w')

def LinkFile(src, dest, nosrcok = False):
    if os.path.abspath(src) == os.path.abspath(dest): return
    if os.path.exists(src):
        # Remove broken link
        if os.path.islink(dest) and not os.path.exists(dest):
            os.remove(dest)
            os.symlink(src, dest)
        elif os.path.exists(dest):
            # Todo: Do we really want to keep an existing symbolic link if it might point to a different file?
            if os.path.islink(dest): pass
            else:
                logger.error("Tried to create symbolic link %s to %s, destination exists but isn't a symbolic link\n" % (src, dest))
                raise RuntimeError
        else:
            os.symlink(src, dest)
    else:
        if not nosrcok:
            logger.error("Tried to create symbolic link %s to %s, but source file doesn't exist%s\n" % (src,dest,MissingFileInspection(src)))
            raise RuntimeError


def CopyFile(src, dest):
    if os.path.exists(src):
        if os.path.exists(dest):
            if os.path.islink(dest):
                logger.error("Tried to copy %s to %s, destination exists but it's a symbolic link\n" % (src, dest))
                raise RuntimeError
        else:
            shutil.copy2(src, dest)
    else:
        logger.error("Tried to copy %s to %s, but source file doesn't exist%s\n" % (src,dest,MissingFileInspection(src)))
        raise RuntimeError

def link_dir_contents(abssrcdir, absdestdir):
    """ 
    Link the contents of the folder abssrcdir into absdestdir.
    First create absdestdir if it's not already an existing folder.
    Next, remove broken symlinks (but not good links or files) in absdestdir
    and create symlinks from abssrcdir by relative linking.
    """
    
    abssrcdir = os.path.abspath(abssrcdir)
    absdestdir = os.path.abspath(absdestdir)
    if os.path.isfile(absdestdir):
        raise RuntimeError("absdestdir %s is a file\n" % absdestdir)
    elif not os.path.exists(absdestdir):
        os.makedirs(absdestdir)
    relpath = os.path.relpath(abssrcdir, absdestdir)
    cwd = os.getcwd()
    os.chdir(absdestdir)
    for fnm in os.listdir(abssrcdir):
        srcfnm = os.path.join(abssrcdir, fnm)
        # Remove broken symlinks if they exist
        if os.path.islink(fnm) and not os.path.exists(fnm):
            os.remove(fnm)
        if os.path.isfile(srcfnm) or (os.path.isdir(srcfnm) and fnm == 'IC'): 
            # Not sure what the 'IC' is - might remove later
            if not os.path.exists(fnm):
                #print "Linking %s to %s" % (srcfnm, destfnm)
                os.symlink(os.path.join(relpath, fnm), fnm)
    os.chdir(cwd)

def remove_if_exists(fnm):
    """ Remove the file if it exists (doesn't return an error). """
    if os.path.exists(fnm):
        os.remove(fnm)

def which(fnm):
    # Get the location of a file.  Works only on UNIX-like file systems.
    try:
        return os.path.split(os.popen('which %s 2> /dev/null' % fnm).readlines()[0].strip())[0]
    except:
        return ''

def copy_tree_over(src, dest):
    """
    Copy a source directory tree to a destination directory tree,
    overwriting files as necessary.  This does not require removing
    the destination folder, which can reduce the number of times
    shutil.rmtree needs to be called.
    """
    if sys.version_info < (3, 8):
        # From https://stackoverflow.com/questions/9160227/dir-util-copy-tree-fails-after-shutil-rmtree/28055993 :
        # If you copy folder, then remove it, then copy again it will fail, because it caches all the created dirs.
        # To workaround you can clear _path_created before copy:
        distutils.dir_util._path_created = {}
        distutils.dir_util.copy_tree(src, dest, verbose=0)
    else:
        shutil.copytree(src, dest, dirs_exist_ok=True)

# Thanks to cesarkawakami on #python (IRC freenode) for this code.
class LineChunker(object):
    def __init__(self, callback):
        self.callback = callback
        self.buf = ""

    def push(self, data):
        # Added by LPW during Py3 compatibility; ran into some trouble decoding strings such as
        # "a" with umlaut on top.  I guess we can ignore these for now.  For some reason,
        # Py2 never required decoding of data, I can simply add it to the wtring.
        # self.buf += data # Old Py2 code...
        self.buf += data.decode('utf-8')#errors='ignore')
        self.nomnom()

    def close(self):
        if self.buf:
            self.callback(self.buf + "\n")

    def nomnom(self):
        # Splits buffer by new line or carriage return, and passes
        # the splitted results onto processing.
        while "\n" in self.buf or "\r" in self.buf:
            chunk, sep, self.buf = re.split(r"(\r|\n)", self.buf, maxsplit=1)
            self.callback(chunk + sep)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

def _exec(command, print_to_screen = False, outfnm = None, logfnm = None, stdin = "", print_command = True, copy_stdout = True, copy_stderr = False, persist = False, expand_cr=False, print_error=True, rbytes=1, cwd=None, **kwargs):
    """Runs command line using subprocess, optionally returning stdout.
    Options:
    command (required) = Name of the command you want to execute
    outfnm (optional) = Name of the output file name (overwritten if exists)
    logfnm (optional) = Name of the log file name (appended if exists)
    stdin (optional) = A string to be passed to stdin, as if it were typed (use newline character to mimic Enter key)
    print_command = Whether to print the command.
    copy_stdout = Copy the stdout stream; can set to False in strange situations
    copy_stderr = Copy the stderr stream to the stdout stream; useful for GROMACS which prints out everything to stderr (argh.)
    expand_cr = Whether to expand carriage returns into newlines (useful for GROMACS mdrun).
    print_error = Whether to print error messages on a crash. Should be true most of the time.
    persist = Continue execution even if the command gives a nonzero return code.
    rbytes = Number of bytes to read from stdout and stderr streams at a time.  GMX requires rbytes = 1 otherwise streams are interleaved.  Higher values for speed.
    """

    # Dictionary of options to be passed to the Popen object.
    cmd_options={'shell':isinstance(command, six.string_types), 'stdin':PIPE, 'stdout':PIPE, 'stderr':PIPE, 'universal_newlines':expand_cr, 'cwd':cwd}

    # If the current working directory is provided, the outputs will be written to there as well.
    if cwd is not None:
        if outfnm is not None:
            outfnm = os.path.abspath(os.path.join(cwd, outfnm))
        if logfnm is not None:
            logfnm = os.path.abspath(os.path.join(cwd, logfnm))

    # "write to file" : Function for writing some characters to the log and/or output files.
    def wtf(out):
        if logfnm is not None:
            with open(logfnm,'ab+') as f:
                f.write(out.encode('utf-8'))
                f.flush()
        if outfnm is not None:
            with open(outfnm,'wb+' if wtf.first else 'ab+') as f:
                f.write(out.encode('utf-8'))
                f.flush()
        wtf.first = False
    wtf.first = True

    # Preserve backwards compatibility; sometimes None gets passed to stdin.
    if stdin is None: stdin = ""

    if print_command:
        logger.info("Executing process: \x1b[92m%-50s\x1b[0m%s%s%s%s\n" % (' '.join(command) if type(command) is list else command,
                                                               " In: %s" % cwd if cwd is not None else "",
                                                               " Output: %s" % outfnm if outfnm is not None else "",
                                                               " Append: %s" % logfnm if logfnm is not None else "",
                                                               (" Stdin: %s" % stdin.replace('\n','\\n')) if stdin else ""))
        wtf("Executing process: %s%s\n" % (command, (" Stdin: %s" % stdin.replace('\n','\\n')) if stdin else ""))

    cmd_options.update(kwargs)
    p = subprocess.Popen(command, **cmd_options)

    # Write the stdin stream to the process.
    p.stdin.write(stdin.encode('ascii'))
    p.stdin.close()

    #===============================================================#
    #| Read the output streams from the process.  This is a bit    |#
    #| complicated because programs like GROMACS tend to print out |#
    #| stdout as well as stderr streams, and also carriage returns |#
    #| along with newline characters.                              |#
    #===============================================================#
    # stdout and stderr streams of the process.
    streams = [p.stdout, p.stderr]
    # Are we using Python 2?
    p2 = sys.version_info.major == 2
    # These are functions that take chunks of lines (read) as inputs.
    def process_out(read):
        if print_to_screen:
            # LPW 2019-11-25: We should be writing a string, not a representation of bytes
            if p2: sys.stdout.write(str(read.encode('utf-8')))
            else: sys.stdout.write(read)
        if copy_stdout:
            process_out.stdout.append(read)
            wtf(read)
    process_out.stdout = []

    def process_err(read):
        if print_to_screen:
            if p2: sys.stderr.write(str(read.encode('utf-8')))
            else: sys.stderr.write(read)
        process_err.stderr.append(read)
        if copy_stderr:
            process_out.stdout.append(read)
            wtf(read)
    process_err.stderr = []
    # This reads the streams one byte at a time, and passes it to the LineChunker
    # which splits it by either newline or carriage return.
    # If the stream has ended, then it is removed from the list.
    with LineChunker(process_out) as out_chunker, LineChunker(process_err) as err_chunker:
        while True:
            to_read, _, _ = select(streams, [], [])
            for fh in to_read:
                # We want to call fh.read below, but this can lead to a system hang when executing Tinker on mac.
                # This hang can be avoided by running fh.read1 (with a "1" at the end), however python2.7
                # doesn't implement ByteStream.read1. So, to enable python3 builds on mac to work, we pick the "best"
                # fh.read function we can get
                if hasattr(fh, 'read1'): fhread = fh.read1
                else: fhread = fh.read

                if fh is p.stdout:
                    read_nbytes = 0
                    read = ''.encode('utf-8')
                    while True:
                        if read_nbytes == 0:
                            read += fhread(rbytes)
                            read_nbytes += rbytes
                        else:
                            read += fhread(1)
                            read_nbytes += 1
                        if read_nbytes > 10+rbytes:
                            raise RuntimeError("Failed to decode stdout from external process.")
                        if not read:
                            streams.remove(p.stdout)
                            p.stdout.close()
                            break
                        else:
                            try:
                                out_chunker.push(read)
                                break
                            except UnicodeDecodeError: # pragma: no cover
                                pass
                elif fh is p.stderr:
                    read_nbytes = 0
                    read = ''.encode('utf-8')
                    while True:
                        if read_nbytes == 0:
                            read += fhread(rbytes)
                            read_nbytes += rbytes
                        else:
                            read += fhread(1)
                            read_nbytes += 1
                        if read_nbytes > 10+rbytes:
                            raise RuntimeError("Failed to decode stderr from external process.")
                        if not read:
                            streams.remove(p.stderr)
                            p.stderr.close()
                            break
                        else:
                            try:
                                err_chunker.push(read)
                                break
                            except UnicodeDecodeError: # pragma: no cover
                                pass
                else:
                    raise RuntimeError
            if len(streams) == 0: break

    p.wait()

    process_out.stdout = ''.join(process_out.stdout)
    process_err.stderr = ''.join(process_err.stderr)

    _exec.returncode = p.returncode
    if p.returncode != 0:
        if process_err.stderr and print_error:
            logger.warning("Received an error message:\n")
            logger.warning("\n[====] \x1b[91mError Message\x1b[0m [====]\n")
            logger.warning(process_err.stderr)
            logger.warning("[====] \x1b[91mEnd o'Message\x1b[0m [====]\n")
        if persist:
            if print_error:
                logger.info("%s gave a return code of %i (it may have crashed) -- carrying on\n" % (command, p.returncode))
        else:
            # This code (commented out) would not throw an exception, but instead exit with the returncode of the crashed program.
            # sys.stderr.write("\x1b[1;94m%s\x1b[0m gave a return code of %i (\x1b[91mit may have crashed\x1b[0m)\n" % (command, p.returncode))
            # sys.exit(p.returncode)
            logger.error("\x1b[1;94m%s\x1b[0m gave a return code of %i (\x1b[91mit may have crashed\x1b[0m)\n\n" % (command, p.returncode))
            raise RuntimeError

    # Return the output in the form of a list of lines, so we can loop over it using "for line in output".
    Out = process_out.stdout.split('\n')
    if Out[-1] == '':
        Out = Out[:-1]
    return Out
_exec.returncode = None

def warn_press_key(warning, timeout=10): # pragma: no cover
    logger.warning(warning + '\n')
    if sys.stdin.isatty():
        logger.warning("\x1b[1;91mPress Enter or wait %i seconds (I assume no responsibility for what happens after this!)\x1b[0m\n" % timeout)
        try:
            rlist, wlist, xlist = select([sys.stdin], [], [], timeout)
            if rlist:
                sys.stdin.readline()
        except: pass

def warn_once(warning, warnhash = None): # pragma: no cover
    """ Prints a warning but will only do so once in a given run. """
    if warnhash is None:
        warnhash = warning
    if warnhash in warn_once.already:
        return
    warn_once.already.add(warnhash)
    if type(warning) is str:
        logger.info(warning + '\n')
    elif type(warning) is list:
        for line in warning:
            logger.info(line + '\n')
warn_once.already = set()
