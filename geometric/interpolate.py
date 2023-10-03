#!/usr/bin/env python

import sys
sys.setrecursionlimit(10000)
from copy import deepcopy
import json
import numpy as np
import networkx as nx
import scipy.special
from geometric.params import InterpParams, parse_interpolate_args
from geometric.molecule import *
from geometric.internal import *
from geometric.nifty import ang2bohr, logger, commadash
from geometric.step import calc_drms_dmax

# Pretend the "memory leaks likely.." has been printed already
CacheWarning = True

# Enable logging output from geomeTRIC packages
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

# Uncomment the below 3 lines to activate geomeTRIC module logging
logger.setLevel(INFO)
handler = RawStreamHandler()
logger.addHandler(handler)

def write_coord_segment(M, coord_segment, output_filename):
    M_copy = deepcopy(M)
    M_copy.xyzs = [x.reshape(-1, 3)*bohr2ang for x in coord_segment]
    M_copy.write(output_filename)

# Need to refactor this so that it uses the minimum of the equilibrium bond length and the sum of covalent radii.
def get_rab(M, i, j, bohr=True):
    if bohr:
        return (Radii[Elements.index(M.elem[i])-1] + Radii[Elements.index(M.elem[j])-1])*ang2bohr
    else:
        return (Radii[Elements.index(M.elem[i])-1] + Radii[Elements.index(M.elem[j])-1])

def find_transfers_common_bonds(molecule, allow_larger=False):
    M = molecule
    transfers = []
    transfer_bonds = []
    elem = M.elem
    natom = len(elem)

    reac = nx.Graph()
    prod = nx.Graph()
    common = nx.Graph()

    M_reac = M[0]
    M_prod = M[-1]
    M_reac.build_topology()
    M_prod.build_topology()

    # Adding atoms
    for i in range(natom):
        reac.add_node(i)
        prod.add_node(i)
        common.add_node(i)

    # Adding bonds for the reactant and product
    for (i, j) in M_reac.bonds:
        reac.add_edge(i, j)

    for (i, j) in M_prod.bonds:
        prod.add_edge(i, j)

    # Adding common bonds
    for edge in reac.edges:
        if prod.has_edge(*edge):
            common.add_edge(*edge)

    # Sizes of the fragments that are intact throughout the reaction
    atom_to_fragment_size = {}
    for c in nx.connected_components(common):
        for i in c:
            atom_to_fragment_size[i] = len(c)

    # Detecting bond donors and acceptors
    for i in range(natom):
        bonds_init = set(nx.neighbors(reac, i))
        bonds_final = set(nx.neighbors(prod, i))
        potential_donors = bonds_init - bonds_final
        potential_acceptors = bonds_final - bonds_init
        # If one bond breaks and one bond forms:
        if len(potential_donors) == 1 and len(potential_acceptors) == 1:
            don = list(potential_donors)[0]
            acc = list(potential_acceptors)[0]
            if allow_larger or (atom_to_fragment_size[i] <= atom_to_fragment_size[don] and atom_to_fragment_size[i] <= atom_to_fragment_size[acc]):
            #if True:
                transfers.append((don, i, acc))
                transfer_bonds.append((min(don, i), max(don, i)))
                transfer_bonds.append((min(acc, i), max(acc, i)))
    return transfers, transfer_bonds, list(common.edges)

def find_nonbonded_pairs(molecule):
    M = molecule
    union_bonds = set()
    nbpairs = []
    elem = M.elem
    natom = len(elem)

    # Detecting all bonds
    for i in [0, -1]:
        Mi = M[i]
        Mi.build_topology()
        for i, j in Mi.bonds:
            union_bonds.add((i, j))

    # Saving the bonds that weren't detected (non-bonded pairs)
    for i in range(natom):
        for j in range(i+1, natom):
            if (i, j) not in union_bonds:
                nbpairs.append((i, j))
    return sorted(nbpairs), sorted(list(union_bonds))

def find_blocks(mtx, thre=1):
    # This is for the diagnostic map
    blocks = []
    block_rights = []
    for i in range(mtx.shape[0]):
        for j in range(i+1, mtx.shape[0]):
            if not (mtx[i:j, i:j]<=thre).all(): 
                if j not in block_rights:
                    blocks.append((i, j-1))
                    block_rights.append(j)
                break
            elif j == mtx.shape[0]-1:
                if j not in block_rights:
                    blocks.append((i, j))
                    block_rights.append(j)
    print(blocks)

def find_path(mtx, thre=1):
    # Converts the diagnostic map into a number of segments.
    # These should be overlapping segments of frame numbers
    # where a single IC is known to describe them all well.
    n = mtx.shape[0]
    # Making a matrix that is an "IC maze"
    okay = (mtx <= thre).astype(int)
    start_ic = 0
    allowed_IC = [[] for i in range(n)]
    mtx_tried = np.zeros_like(mtx)

    # If the starting point or end point is blocked, we can't escape.
    if not okay[n-1, n-1] or not okay[0, 0]:
        raise RuntimeError("Spoo!")

    okay2 = okay.copy()
    for niter in range(n**2):
        for ic in range(0, n):
            for ix in range(0, n):
                if (ic == n-1 and ix == n-1) or (ic == 0 and ix == 0): # Do nothing at the starting point and end point
                    pass
                elif ic == n-1 and not okay[ic, ix+1]: # Lower right corner on bottom
                    okay2[ic, ix] = 0
                # elif ix == n-1 and not okay[ic+1, ix]: # Lower right corner on right side
                #     okay2[ic, ix] = 0
                # elif ix == 0 and not okay[ic-1, ix]:  # Upper left corner on left side
                #     okay2[ic, ix] = 0
                elif ic == 0 and not okay[ic, ix-1]:  # Upper left corner on top side
                    okay2[ic, ix] = 0
                elif (ic<(n-1) and not okay[ic+1, ix]) and (ix<(n-1) and not okay[ic, ix+1]):
                    okay2[ic, ix] = 0
                elif (ic>0 and not okay[ic-1, ix]) and (ix>0 and not okay[ic, ix-1]):
                    okay2[ic, ix] = 0
        ndiff = np.sum(okay2 != okay)
        okay = okay2.copy()
        if ndiff == 0: break # Break when the maze is "converged"

    print_map(okay, "Allowed driving regions", colorscheme=1)

    def valid(i, j, size):
        i0=max(0,i-size)
        i1=min(n,i+size+1)
        j0=max(0,j-size)
        j1=min(n,j+size+1)
        return okay[i0:i1,j0:j1].all()

    segments_by_size = {}

    verbose = 0
    for size in range(n//2):
        if verbose >= 1:
            print("Trying size %i (grid size %i)" % (size, 2*size+1))
        # Start from the left side..
        for ic0 in range(0, n):
            segments = []
            ic = ic0
            ix = 0
            steps = 0
            if not okay[ic, ix]: continue
            ix_last = 0
            ic_ix_mode = 0
            if verbose >= 1:
                print("From Left:   IC %i Size %i" % (ic, size))
            while True:
                steps += 1
                if ix == n-1: # or ic == n-1: 
                    if verbose >= 2:
                        print("Reached the end! (%i,%i)" % (ic, ix))
                    segments.append((ix_last, n-1))
                    if verbose >= 1:
                        print("Found segments: ", segments)
                    if size not in segments_by_size:
                        segments_by_size[size] = segments[:]
                    else:
                        if len(segments) <= len(segments_by_size[size]):
                            shortest_segment = min([j-i for (i, j) in segments])
                            shortest_segment_stored = min([j-i for (i, j) in segments_by_size[size]])
                            if shortest_segment > shortest_segment_stored:
                                segments_by_size[size] = segments[:]
                    success = True
                    break
                elif not valid(ic, ix, size) or steps > 2*n:
                    if verbose >= 2:
                        print("Dead end at (%i,%i)" % (ic, ix))
                    success = False
                    break
                if ic_ix_mode == 0:
                    ix += 1
                    if not valid(ic, ix, size):
                        ix -= 1
                        ic_ix_mode = 1
                        if verbose >= 2:
                            print("Changed to ic-direction at (%i,%i)" % (ic, ix))
                        segments.append((ix_last, ix))
                        ix_last = ix + 1
                        ic += 1
                else:
                    ic += 1
                    if ic == n or not valid(ic, ix, size):
                        ic -= 1
                        ic_ix_mode = 0
                        if verbose >= 2:
                            print("Changed to ix-direction at (%i,%i)" % (ic, ix))
                        ix += 1
        # Start from the right side.
        for ic0 in list(range(0, n))[::-1]:
            segments = []
            ic = ic0
            ix = n-1
            steps = 0
            if not okay[ic, ix]: continue
            ix_last = n-1
            ic_ix_mode = 0
            if verbose >= 1:
                print("From Right:  IC %i Size %i" % (ic, size))
            while True:
                steps += 1
                if ix == 0: # or ic == 0: 
                    if verbose >= 2:
                        print("Reached the end! (%i,%i)" % (ic, ix))
                    segments.append((0, ix_last))
                    segments = segments[::-1]
                    if verbose >= 1:
                        print("Found segments: ", segments)
                    if size not in segments_by_size:
                        segments_by_size[size] = segments[:]
                    else:
                        if len(segments) <= len(segments_by_size[size]):
                            shortest_segment = min([j-i for (i, j) in segments])
                            shortest_segment_stored = min([j-i for (i, j) in segments_by_size[size]])
                            if shortest_segment > shortest_segment_stored:
                                segments_by_size[size] = segments[:]
                    success = True
                    break
                elif not valid(ic, ix, size) or steps > 2*n:
                    if verbose >= 2:
                        print("Dead end at (%i,%i)" % (ic, ix))
                    success = False
                    break
                if ic_ix_mode == 0:
                    ix -= 1
                    if not valid(ic, ix, size):
                        ix += 1
                        ic_ix_mode = 1
                        if verbose >= 2:
                            print("Changed to ic-direction at (%i,%i)" % (ic, ix))
                        segments.append((ix, ix_last))
                        ix_last = ix - 1
                        ic -= 1
                else:
                    ic -= 1
                    if ic == -1 or not valid(ic, ix, size):
                        ic += 1
                        ic_ix_mode = 0
                        if verbose >= 2:
                            print("Changed to ix-direction at (%i,%i)" % (ic, ix))
                        ix -= 1
        # if not success:
        #     print("No segments found at this size")
    if segments_by_size:
        min_pathsize = None
        largest_size = None
        for key in sorted(list(segments_by_size.keys())):
            if min_pathsize is None:
                largest_size = key
                min_pathsize = len(segments_by_size[key])
            if len(segments_by_size[key]) == min_pathsize:
                largest_size = key
        print("Chose these segments (car size %i):" % largest_size)
        keep_segments = segments_by_size[largest_size]
        print(keep_segments)
    else:
        raise RuntimeError("Path finding algorithm failed")
        # if success:
        #     if keep_segments and len(segments) > len(keep_segments):
        #         print("Keeping the *previous* segments as there were fewer of them")
        #         break
        #     else:
        #         print("Obtained segments at grid size %i:" % (2*size+1), segments)
        #         keep_segments = segments[:]
    return keep_segments

def merge_one_pair(segments, imerge):
    merged = []
    for i in range(len(segments)):
        if i == imerge:
            merged.append((segments[i][0], segments[i+1][1]))
        elif i == imerge + 1: 
            continue
        else:
            merged.append((segments[i]))
    return merged

def fill_splice_segments(segments, splice_length):
    n_frames = segments[-1][-1] + 1
    filled = segments[:]
    spliced = []
    while min([s[1]-s[0] for s in filled]) < splice_length:
        merge_lengths = []
        for imerge in range(len(filled)-1):
            merged = merge_one_pair(filled, imerge)
            merge_lengths.append(max([s[1]-s[0] for s in merged]))
        imerge_best = np.argmin(merge_lengths)
        filled = merge_one_pair(filled, imerge_best)
    for i, j in filled:
        i = max(0, i-splice_length//2)
        j = min(n_frames-1, j+splice_length//2)
        spliced.append((i, j))
    # Sometimes the first or last segment is too short
    if len(spliced) > 1 and spliced[1][0] == 0:
        spliced = spliced[1:]
        filled = filled[1:]
        filled[0] = (0, filled[0][1])
    if len(spliced) > 1 and  spliced[-2][1] == n_frames-1:
        spliced = spliced[:-1]
        filled = filled[:-1]
        filled[-1] = (filled[-1][0], n_frames-1)
    return filled, spliced

def splice_segment_endpoints(coord_segments, segment_endpoints, segments_spliced, splice_length, damping, verbose=False):
    max_mismatch = 0.0
    segment_endpoints = deepcopy(segment_endpoints)
    for a in range(len(segments_spliced)-1):
        b = a + 1
        xyz_a_spliceStart = coord_segments[a][-splice_length]
        xyz_b_spliceStart = coord_segments[b][0]
        xyz_a_spliceEnd = coord_segments[a][-1]
        xyz_b_spliceEnd = coord_segments[b][splice_length-1]
        _, dmax1 = calc_drms_dmax(xyz_b_spliceStart, xyz_a_spliceStart, align=False)
        _, dmax2 = calc_drms_dmax(xyz_b_spliceEnd, xyz_a_spliceEnd, align=False)
        if verbose: print("Segment %i-%i splice mismatches: %.3e %.3e" % (a, b, dmax1, dmax2))
        max_mismatch = max(max_mismatch, max(dmax1, dmax2))
        segment_endpoints[a][1] = damping*xyz_b_spliceEnd.copy() + (1.0-damping)*xyz_a_spliceEnd.copy()
        segment_endpoints[b][0] = damping*xyz_a_spliceStart.copy() + (1.0-damping)*xyz_b_spliceStart.copy()
    return segment_endpoints, max_mismatch

def dq_scale_prims(IC, dest_coords, curr_coords, scale_factors={}, sync=0):
    # Calculate the differences in primitive coordinates.
    dqPrims = IC.Prims.calcDiff(dest_coords, curr_coords, sync=sync)
    for i, Prim in enumerate(IC.Prims.Internals):
        # Scale each primitive by the corresponding scale factor for that type
        if type(Prim) in [RotationA, RotationB, RotationC]:
            dqPrims[i] *= scale_factors.get('rotation', 0.0)
        elif type(Prim) in [Dihedral]:
            # Dihedrals that aren't "proper" have their dq's set to zero
            bond1 = Distance(Prim.a, Prim.b)
            bond2 = Distance(Prim.b, Prim.c)
            bond3 = Distance(Prim.c, Prim.d)
            ang1 = Angle(Prim.a, Prim.b, Prim.c)
            ang2 = Angle(Prim.b, Prim.c, Prim.d)
            if any([np.abs(bond.value(dest_coords) - bond.value(curr_coords)) > 0.5 for bond in [bond1, bond2, bond3]]): dqPrims[i] *= 0.0
            for coords in [curr_coords, dest_coords]:
                if np.abs(np.cos(ang1.value(coords))) > 0.7: dqPrims[i] *= 0.0
                if np.abs(np.cos(ang2.value(coords))) > 0.7: dqPrims[i] *= 0.0
            dqPrims[i] *= scale_factors.get('dihedral', 0.0)
        elif type(Prim) in [TranslationX, TranslationY, TranslationZ, CartesianX, CartesianY, CartesianZ]:
            dqPrims[i] *= scale_factors.get('translation', 0.0)
        elif type(Prim) is Distance and Prim.n < 0:
            # Experimental; inverse distances are scaled to one-half of their destination values
            thre = 0.5
            if Prim.value(dest_coords) > thre:
                dqPrims[i] = thre - Prim.value(curr_coords)
        else:
            dqPrims[i] *= 0.0
    dq = np.dot(dqPrims, IC.Vecs)
    return dq

def interpolate_segment(IC, curr_coords, dest_coords, nDiv, backward=False, rebuild_dlc=False, err_thre=1e-2, verbose=0):
    if backward:
        tmp = curr_coords.copy()
        curr_coords = dest_coords.copy()
        dest_coords = tmp.copy()
    dq = IC.calcDiff(dest_coords, curr_coords, sync=1)
    coord_segment = [curr_coords]
    for k in range(nDiv):
        if rebuild_dlc:
            IC.build_dlc(curr_coords)
            dq = IC.calcDiff(dest_coords, curr_coords, sync=1)
            new_coords = IC.newCartesian(curr_coords, dq/(nDiv-k), verbose=verbose)
        else:
            new_coords = IC.newCartesian(curr_coords, dq/nDiv, verbose=verbose)
        coord_segment.append(new_coords)
        curr_coords = new_coords.copy()
    _, endpt_err = calc_drms_dmax(curr_coords, dest_coords, align=True)
    if backward:
        coord_segment = coord_segment[::-1]
    success = endpt_err < err_thre
    return coord_segment, endpt_err, success

def get_segment_endpoints(M, segments_spliced):
    segment_endpoints = []
    for a, (i, j) in enumerate(segments_spliced):
        xyzi = M.xyzs[i].flatten() * ang2bohr
        xyzj = M.xyzs[j].flatten() * ang2bohr
        segment_endpoints.append(np.array([xyzi.copy(), xyzj.copy()]))
    return segment_endpoints

def split_segments(segments_spliced, fail_segment, splice_length):
    segments_filled_new = []
    segments_spliced_new = []
    n = len(segments_spliced)

    for a, (i, j) in enumerate(segments_spliced):
        ii = i if a == 0 else i + splice_length//2
        jj = j if a == n-1 else j - splice_length//2
        if a == fail_segment:
            if (j-i) < splice_length:
                raise RuntimeError("Failed to interpolate a segment that's shorter than the splice length")
            midpt = (i+j)//2
            segments_filled_new.append((ii, midpt-1))
            segments_filled_new.append((midpt, jj))
            segments_spliced_new.append((i, midpt-1+splice_length//2))
            segments_spliced_new.append((midpt-splice_length//2, j))
        else:
            segments_filled_new.append((ii, jj))
            segments_spliced_new.append((i, j))
        # i = max(0, i-splice_length//2)
        # j = min(len(M)-1, j+splice_length//2)
        # segments_spliced.append((i, j))
    print("Split segment %i - now the segments are:" % fail_segment)
    print("Without splices:", segments_filled_new)
    print("With splices:", segments_spliced_new)
    return segments_filled_new, segments_spliced_new

def print_map(mtx, title, colorscheme=0):
    print(title)
    n = mtx.shape[0]
    for i in range(n):
        if i == 0:
            print(" x: ", end='')
            for j in range(n):
                print("%2i" % j, end='')
            print()
        print("IC%2i " % i, end='')
        for j in range(n):
            value = mtx[i, j]
            if colorscheme == 0:
                if value == 0: color = '\x1b[94m'
                elif value == 1: color = '\x1b[92m'
                elif value >= 2: color = '\x1b[91m'
            elif colorscheme == 1:
                if value == 0: color = '\x1b[91m'
                elif value == 1: color = '\x1b[92m'
            else: raise RuntimeError("Invalid color scheme")
            print("%s%1i\x1b[0m " % (color, value), end='')
        print("IC%2i " % i, end='')
        print()
        if i == n-1:
            print(" x: ", end='')
            for j in range(n):
                print("%2i" % j, end='')
            print()

class Interpolator(object):
    def __init__(self, M_in, n_frames = 50, use_midframes = False, align_system=False, do_prealign=False, verbose=0):
        # The input molecule; it should not be modified.
        self.M_in = deepcopy(M_in)
        # Check the length of the input molecule
        assert len(self.M_in) >= 2
        if use_midframes:
            self.do_init = False
            assert len(self.M_in) >= 5, "When setting use_midframes, input molecule must have >= 5 structures"
            assert do_prealign is False, "Do not pass do_prealign if using intermediate frames"
            assert n_frames == 0, "Do not pass n_frames when setting use_midframes"
            self.n_frames = len(self.M_in)
        else:
            self.do_init = True
            self.n_frames = n_frames
        assert self.n_frames >= 5, "Number of frames for interpolation must be >= 5"
        # Determine the splice length; currently restricted to a number between 2 and 6
        self.get_splice_length()
        # The Molecule object that is currently being processed.
        self.M = deepcopy(self.M_in)
        self.n_atoms = len(self.M.elem)
        # Whether to pre-align the fragments in the reactant and product structure
        self.do_prealign = do_prealign
        # Whether to align the input frames to the reactant structure upon input and before writing output
        self.align_system = True if self.do_prealign else align_system
        if self.align_system:
            self.M.align()
        # Atom pairs that are not bonded (resp. bonded) in either the reactant or product
        self.nbpairs, self.union_bonds = find_nonbonded_pairs(self.M)
        # List of (donor, transfer, acceptor) triplets.
        # List of atom pairs in bonds that are involved in transfer triplets.
        # List of atom pairs that are bonded in BOTH the reactant and product
        self.transfers, self.transfer_bonds, self.common_bonds = find_transfers_common_bonds(self.M)
        print("Found these transfers:", self.transfers)
        # G matrix condition numbers.
        self.Gcond_matrix = np.zeros((self.n_frames, self.n_frames), dtype=float)
        # Molecule objects containing the endpoints.
        # endmols_in contains the path endpoints before any pre-alignment of fragments.
        self.endmols_in = [deepcopy(self.M[0]), deepcopy(self.M[-1])]
        self.endmols = deepcopy(self.endmols_in)
        self.endbonds = []
        self.endfrags = []
        self.endxyzs = []
        self.enddists = []
        for mol in self.endmols:
            mol.build_topology()
            self.endbonds.append(mol.bonds[:])
            self.endfrags.append(len(mol.molecules))
            self.endxyzs.append(mol.xyzs[0].flatten()*ang2bohr)
            atom_pairs, distance_matrix = mol.distance_matrix(pbc=False)
            self.enddists.append(distance_matrix[0].copy())
        # List of atom pairs from Molecule.distance_matrix()
        # but with np.int32 converted to int
        self.atom_pairs = [list(pair) for pair in atom_pairs.copy()]
        # The smaller of the distances at either endpoint for each pair
        self.min_enddists = np.min(np.array(self.enddists), axis=0)
        self.verbose = verbose

    def get_splice_length(self):
        # Determine the splice length
        splice_length = int(np.round(self.n_frames/10))
        splice_length += splice_length % 2
        if splice_length > 6: splice_length = 6
        if splice_length < 2: splice_length = 2
        self.splice_length = splice_length

    def new_molecule(self, coord_list, comms=None):
        M = deepcopy(self.M[0])
        M.xyzs = [x.reshape(-1, 3)*bohr2ang for x in coord_list]
        if comms is not None:
            M.comms = comms
        else:
            M.comms = ['generated by geometric-interpolate; frame %i' % i for i in range(len(coord_list))]
        return M

    def get_clash_thresholds(self, addthre=0.6, altdists=None, altthre=0.9):
        R = []
        for pairidx, (i, j) in enumerate(self.atom_pairs):
            rab = get_rab(self.M, i, j, bohr=False)
            if altdists is not None:
                R.append(min(max(1.0, rab)+addthre, altdists[pairidx]*altthre))
            else:
                R.append(max(1.0, rab)+addthre)
        R = np.array(R)
        return R

    def detect_clash_one_frame(self, coords, addthre=0.6, altdists=None, altthre=0.9):
        # Detect clashes for input coordinates provided in Bohr.
        assert type(coords) is np.ndarray
        M = deepcopy(self.M[0])
        M.xyzs = [coords.reshape(-1, 3)*bohr2ang]
        atom_pairs = self.atom_pairs
        _, distance_matrix = M.distance_matrix(pbc=False)
        R = self.get_clash_thresholds(addthre=addthre, altdists=altdists, altthre=altthre)

        clash_pairs = []
        for pairidx, close in enumerate(distance_matrix[0] < R):
            if not close: continue
            i = self.atom_pairs[pairidx][0]
            j = self.atom_pairs[pairidx][1]
            assert i < j
            if (i, j) not in clash_pairs:
                clash_pairs.append((int(i), int(j)))
        return clash_pairs

    def detect_clash_trajectory(self, coords, clash_known=[], addthre=0.6, altdists=None, altthre=0.9, verbose=True):
        # Detect clashes for list of input coordinates provided in Bohr.
        assert type(coords) is list
        assert type(coords[0]) is np.ndarray
        M = deepcopy(self.M[0])
        M.xyzs = [x.reshape(-1, 3)*bohr2ang for x in coords]
        atom_pairs = self.atom_pairs
        _, distance_matrix = M.distance_matrix(pbc=False)
        R = self.get_clash_thresholds(addthre=addthre, altdists=altdists, altthre=altthre)

        clash_known = deepcopy(clash_known)
        clash_new = []
        clash_frames = {}
        for k in range(len(M)):
            for pairidx, close in enumerate(distance_matrix[k] < R):
                if not close: continue
                i = atom_pairs[pairidx][0]
                j = atom_pairs[pairidx][1]
                assert i < j
                if (i, j) not in clash_known:
                    clash_known.append((int(i), int(j)))
                    clash_new.append((int(i), int(j)))
                clash_frames.setdefault(pairidx, []).append(k)

        for pairidx, frames in clash_frames.items():
            i = atom_pairs[pairidx][0]
            j = atom_pairs[pairidx][1]
            min_dist = np.min(np.array(distance_matrix)[np.array(frames),pairidx])
            if verbose:
                print(">> Clash %10s: %8s at frames %6s (%9s, closest = %.3f, %.3f of thre)" % ("(new)" if (i, j) in clash_new else "(existing)",
                                                                                                "%s%i-%s%i" % (M.elem[i], i+1, M.elem[j], j+1), 
                                                                                                commadash(frames), "bonded" if (i, j) in self.union_bonds else "nonbonded",
                                                                                                min_dist, min_dist/R[pairidx]))
        return clash_known, clash_new

    def prealign_one_stage(self, curr_coords, dest_coords, nDiv, scale_factors={'dihedral':0.0, 'translation':0.0, 'rotation':0.0}):
        IC0, IC1 = self.endICs
        n_frag0, n_frag1 = self.endfrags
        if n_frag0 == 1 and n_frag1 == 1:
            coord_segment0 = [curr_coords]
            coord_segment1 = [dest_coords]
            coord_segment = [curr_coords, dest_coords]
            return coord_segment, coord_segment0, coord_segment1
        
        # The formula here is a bit complicated. Basically, if scale_factors[key]
        # is set to a quantity between 0 and 1, we want the prealignment to bring 
        # the endpoints together only part-way. However, if we simply multiply the dq 
        # of primitives by the scale factor, the remaining distance is still large 
        # by the time we get to the middle and the dividing "n" becomes one, causing larger steps 
        # to be taken than intended. Therefore, for each step taken, we calculate how 
        # far we have moved toward the middle on the "number line", and change the
        # scale factor accordingly so the next step takes us the same distance down
        # the number line.
        scale_factors = deepcopy(scale_factors)
        numerators = {}
        for key, val in scale_factors.items():
            numerators[key] = val/nDiv

        for trial in range(2):
            # First trial - Attempt to bring the selected DoFs of the fragments together completely and keep the result if the final frame has no clashes.
            # Second trial - Each side stops when it encounters a clash and exits when fragments are brought together or clashes found for both.
            coord_segment0 = [curr_coords.copy()]
            coord_segment1 = [dest_coords.copy()]
            IC0.clearCache()
            IC1.clearCache()
            clash0 = []
            clash1 = []
            k = 0
            nstep0 = 0
            nstep1 = 0
            distances = {'dihedral':1.0, 'translation':1.0, 'rotation':1.0}
            while True:
                if n_frag0 > 1 and (trial==0 or not clash0):
                    # IC0.build_dlc(grow0)
                    dq0 = dq_scale_prims(IC0, coord_segment1[-1], coord_segment0[-1], scale_factors, 1)
                    n = nDiv-k
                    new_coord = IC0.newCartesian(coord_segment0[-1], dq0/n, verbose=0)
                    clash0 = self.detect_clash_one_frame(new_coord, altdists=self.enddists[0])
                    # The codes in this if statement are executed only if this end is still growing
                    if trial==0 or not clash0:
                        for key, val in scale_factors.items(): 
                            distances[key] *= 1-val/n
                            scale_factors[key] = numerators[key]/(distances[key]/(n-1)) if n > 1 else scale_factors[key]
                        coord_segment0.append(new_coord.copy())
                        nstep0 += 1
                        k += 1
                    # if clash0:
                    #     print("Trial %i: Clash in step %i in reactant direction:" % (trial, nstep0), clash0)
                if trial==1 and (clash0 and (clash1 or n_frag1 == 1)): break
                if k >= nDiv: break
                if n_frag1 > 1 and (trial==0 or not clash1):
                    # IC1.build_dlc(grow1)
                    dq1 = dq_scale_prims(IC1, coord_segment1[-1], coord_segment0[-1], scale_factors, 1)
                    dq1 *= -1
                    n = nDiv-k
                    new_coord = IC1.newCartesian(coord_segment1[-1], dq1/n, verbose=0)
                    clash1 = self.detect_clash_one_frame(new_coord, altdists=self.enddists[1])
                    # The codes in this if statement are executed only if this end is still growing
                    if trial==0 or not clash1:
                        for key, val in scale_factors.items(): 
                            distances[key] *= 1-val/n
                            scale_factors[key] = numerators[key]/(distances[key]/(n-1)) if n > 1 else scale_factors[key]
                        coord_segment1.append(new_coord.copy())
                        nstep1 += 1
                        k += 1
                    # if clash1:
                    #     print("Trial %i: Clash in step %i in product direction:" % (trial, nstep1), clash1)
                if trial==1 and (clash1 and (clash0 or n_frag0 == 1)): break
                if k >= nDiv: break
            if trial==0 and not clash0 and not clash1:
                break
    
        coord_segment = coord_segment0 + coord_segment1[::-1][1:]
        return coord_segment, coord_segment0, coord_segment1
    
    def prealign_fragments(self):
        print(">> Aligning molecules in reactant and product")
        # Build ICs for alignment
        M0, M1 = self.endmols
        xyz0, xyz1 = self.endxyzs
        xyz0_stage = xyz0.copy()
        xyz1_stage = xyz1.copy()
        IC0 = DelocalizedInternalCoordinates(M0, build=True, connect=False, addcart=False, connect_isolated=False)
        IC1 = DelocalizedInternalCoordinates(M1, build=True, connect=False, addcart=False, connect_isolated=False)
        self.endICs = [IC0, IC1]
        M_reac = None
        M_prod = None
        for stage in [0, 1]:
            if stage == 0: 
                print("Pre-alignment stage 0: Rotations")
                scale_factors = {'dihedral':0.0, 'translation':0.0, 'rotation':1.0}
            elif stage == 1:
                print("Pre-alignment stage 1: Translations")
                scale_factors = {'dihedral':0.0, 'translation':1.0, 'rotation':0.0}
            coord_segment, segment_reac, segment_prod = self.prealign_one_stage(xyz0_stage, xyz1_stage, self.n_frames, scale_factors)
            M_reac_stage = self.new_molecule(segment_reac)
            M_prod_stage = self.new_molecule(segment_prod)
            M_reac = M_reac_stage if M_reac is None else M_reac + M_reac_stage
            M_prod = M_prod_stage if M_prod is None else M_prod + M_prod_stage
            xyz0_stage = segment_reac[-1].copy()
            xyz1_stage = segment_prod[-1].copy()

        M_reac_frames = EqualSpacing(M_reac, frames=self.n_frames//2+1, RMSD=False, spline=True)
        M_prod_frames = EqualSpacing(M_prod, frames=self.n_frames//2+1, RMSD=False, spline=True)
        M_reac_dx = EqualSpacing(M_reac, dx=0.1, RMSD=False, spline=True)
        M_prod_dx = EqualSpacing(M_prod, dx=0.1, RMSD=False, spline=True)
        # Keep the shorter of the dmax=0.1 or n_frames//2 segments
        M_reac = M_reac_dx if len(M_reac_dx) < len(M_reac_frames) else M_reac_frames
        M_prod = M_prod_dx if len(M_prod_dx) < len(M_prod_frames) else M_prod_frames
        M_reac.comms = ["Alignment for reactant segment; frame %i" % i for i in range(len(M_reac))]
        M_prod.comms = ["Alignment for product segment; frame %i" % i for i in range(len(M_prod))]
        if len(M_reac) > 1:
            if len(M_prod) > 1:
                print("Alignment produced %i additional (reactant) and %i (product) frames" % (len(M_reac)-1, len(M_prod)-1))
            else:
                print("Alignment produced %i additional (reactant) frames" % (len(M_reac)-1))
        elif len(M_prod) > 1:
            print("Alignment produced %i additional (product) frames" % (len(M_prod)-1))
        else:
            print("Alignment produced no additional frames")
        # The endpoints after prealingment.
        M_end = M_reac[-1] + M_prod[-1]
        # The product segment goes from the product back to initial. Reverse it here.
        M_prod = M_prod[::-1]
        if len(M_reac) > 1 or len(M_prod) > 1:
            (M_reac+M_prod).write('prealign.xyz')
        # Overwrite some variables that later routines will use.
        self.M = M_end
        self.endxyzs = []
        self.enddists = []
        self.endmols = [deepcopy(M_reac), deepcopy(M_prod)]
        for mol in self.endmols:
            self.endxyzs.append(mol.xyzs[-1].flatten()*ang2bohr)
            atom_pairs, distance_matrix = mol.distance_matrix(pbc=False)
            self.enddists.append(distance_matrix[-1].copy())

    def initial_guess_one_method(self, method = 0):
        # Class variables that are used in this method
        M = deepcopy(self.M)
        n_frames = self.n_frames
        common_bonds = self.common_bonds
        union_bonds = self.union_bonds
        n_atoms = self.n_atoms

        # if method%2 == 1:
        #     M = M[::-1]
        M0 = M[0]
        M1 = M[-1]
        xyz0 = M.xyzs[0].flatten()*ang2bohr
        xyz1 = M.xyzs[-1].flatten()*ang2bohr
        M_IC = M0 if method%2 == 0 else M1
        xyz_IC = xyz0 if method%2 == 0 else xyz1

        # Build the IC system used to interpolate from one endpoint to the other.
        if method in [0, 1]:
            # Use TRIC coordinate system, but use only the bonds that exist at both endpoints.
            M_IC.bonds = common_bonds
            M_IC.top_settings['read_bonds'] = True
            IC = DelocalizedInternalCoordinates(M_IC, build=True, connect=False, addcart=False, connect_isolated=False)
            newPrims = IC.Prims.Internals
        elif method in [2, 3]:
            # Use HDLC coordinate system, but use only the bonds that exist at both endpoints.
            # Set weight of 0.5 to all Cartesian primitive ICs.
            M_IC.bonds = common_bonds
            M_IC.top_settings['read_bonds'] = True
            IC = DelocalizedInternalCoordinates(M_IC, build=True, connect=False, addcart=False, connect_isolated=False)
            newPrims = IC.Prims.Internals
            for i in range(n_atoms):
                newPrims.append(CartesianX(i, w=0.5))
                newPrims.append(CartesianY(i, w=0.5))
                newPrims.append(CartesianZ(i, w=0.5))
        elif method in [4, 5]:
            # Use all-inverse distances plus Cartesian coordinates with small weight.
            IC = DelocalizedInternalCoordinates(M_IC, build=True, connect=False, addcart=False)
            newPrims = []
            for Prim in IC.Prims.Internals:
                if type(Prim) is Distance:
                    # Include distances that were added automatically (corresponding to bonds)
                    newPrim = Distance(Prim.a, Prim.b, rab=4*get_rab(M, Prim.a, Prim.b), n=-1)
                    if newPrim not in newPrims: newPrims.append(newPrim)
                elif type(Prim) in [Angle, LinearAngle]:
                    # Include distances corresponding to 1-3 atoms of angles
                    newPrim = Distance(Prim.a, Prim.c, rab=4*get_rab(M, Prim.a, Prim.c), n=-1)
                    if newPrim not in newPrims: newPrims.append(newPrim)
                elif type(Prim) is Dihedral:
                    # Include distances corresponding to 1-4 atoms of dihedrals
                    (i, j) = (min(Prim.a, Prim.d), max(Prim.a, Prim.d))
                    newPrim = Distance(i, j, rab=4*get_rab(M, i, j), n=-1)
                    if newPrim not in newPrims: newPrims.append(newPrim)
                elif type(Prim) is OutOfPlane:
                    # Include distances corresponding to b-c, b-d and c-d for out of plane (a is central atom)
                    for (i, j) in [(min(Prim.b, Prim.c), max(Prim.b, Prim.c)),
                                   (min(Prim.b, Prim.d), max(Prim.b, Prim.d)),
                                   (min(Prim.c, Prim.d), max(Prim.c, Prim.d))]:
                        newPrim = Distance(i, j, rab=4*get_rab(M, i, j), n=-1)
                        if newPrim not in newPrims: newPrims.append(newPrim)
            for i in range(n_atoms):
                newPrims.append(CartesianX(i, w=0.25))
                newPrims.append(CartesianY(i, w=0.25))
                newPrims.append(CartesianZ(i, w=0.25))
        else:
            raise RuntimeError("Invalid value for method")

        # Finalize the IC system.
        IC.Prims.Internals = newPrims
        IC.Prims.reorderPrimitives()
        IC.build_dlc(xyz_IC)
        # print("=== Primitives for method %i ===" % method)
        # print(IC.Prims)

        # Build a "checking" IC system used to detect whether any primitive ICs are changing very rapidly between frames.
        # This is one of our diagnostics for whether the generated path is "good".
        M0.bonds = common_bonds
        M0.top_settings['read_bonds'] = True
        IC_Chk = PrimitiveInternalCoordinates(M0, build=True, connect=False, addcart=True, connect_isolated=False)
        M1.bonds = common_bonds
        M1.top_settings['read_bonds'] = True
        IC1_Chk = PrimitiveInternalCoordinates(M1, build=True, connect=False, addcart=True, connect_isolated=False)
        chkPrims = []
        for Prim in IC_Chk.Internals + IC1_Chk.Internals:
            # Keep the primitives on either end that are valid for both points.
            if Prim.diagnostic(xyz0)[0] <= 1 and Prim.diagnostic(xyz1)[0] <= 1 and Prim not in chkPrims:
                chkPrims.append(Prim)
        IC_Chk.Internals = chkPrims
        IC_Chk.reorderPrimitives()

        err_thre = 1e-2
        clash_pairs = []
        nDiv = n_frames - 1
        for clash_round in range(5):
            coord_segment, endpt_err, _ = interpolate_segment(IC, xyz0, xyz1, nDiv, backward=(method%2==1), rebuild_dlc=True)
            dq_segment = np.array([IC_Chk.calcDiff(coord_segment[i], coord_segment[i+1]) for i in range(n_frames-1)])
            # Setting to -1e-6 essentially ignores the checking primitives if they evaluate to +/- inf.
            dq_segment[dq_segment == -np.inf] = -1e-6
            dq_segment[dq_segment == np.inf] = 1e-6
            dq_max = np.max(np.abs(dq_segment), axis=0)
            dq_med = np.median(np.abs(dq_segment), axis=0)
            dq_med[dq_med<0.01] = 0.01
            dq_ratio = max(dq_max/dq_med)
    
            M.xyzs = [x.reshape(-1, 3)*bohr2ang for x in coord_segment]
            M.comms = ['Interpolated frame %i' % k for k in range(len(coord_segment))]
            # Fail if the endpt_err is greater than the threshold
            if endpt_err > err_thre:
                M.align()
                return M, 1, dq_ratio, endpt_err
            # Check for clashes
            clash_pairs, clash_new = self.detect_clash_trajectory(coord_segment, clash_known=clash_pairs, altdists=self.min_enddists, verbose=False)
            if not clash_new:
                # print("No new clashes found; finishing")
                break
            else:
                for (i, j) in clash_pairs:
                    if Distance(i, j) in IC.Prims.Internals: IC.Prims.delete(Distance(i, j))
                    IC.Prims.add(Distance(i, j, rab=4*get_rab(M, i, j), n=-1))
                IC.Prims.reorderPrimitives()
                IC.build_dlc(xyz_IC)
                
        if self.align_system:
            M.align()
        return M, 0, dq_ratio, endpt_err

    def initial_guess(self):
        M = None
        dq_ratio_min = 1e6
        for method in range(6):
            M_, status, dq_ratio, endpt_err = self.initial_guess_one_method(method=method)
            # M_.write('interpolated_endpoints_method%i.xyz' % method)
            # Keep the interpolation path with status=0 and the lowest dq_ratio
            print("Initial pathway generation using method %i %s; dq_ratio = %.3f endpt_err = %.3f" % 
                  (method, "success" if status == 0 else "failed", dq_ratio, endpt_err))
            if status == 0 and dq_ratio < dq_ratio_min:
                M = deepcopy(M_)
                dq_ratio_min = dq_ratio
                keep_method = method
        if M is None:
            raise RuntimeError("Failed to generate an initial path")
        M.comms = [c + " (method %i)" % keep_method for c in M.comms]
        M.write("initial_guess.xyz")
        # Overwrite the Molecule object currently being processed
        self.M = deepcopy(M)
        print("Keeping the result from method %i" % keep_method)

    def assign_ICs_to_segments(self):
        M = self.M
        ICs = self.ICs
        segments_filled = self.segments_filled
        diag_matrix = self.diag_matrix
        Gcond_matrix = self.Gcond_matrix
        n_frames = self.n_frames

        segment_to_ICs = []
        print("Determining which IC to use in each segment:")
        for a, (ii, jj) in enumerate(segments_filled):
            print("=== Now working on segment (%i, %i) ===" % (ii, jj))
            Gcond_maxs = []
            diags = []
            # For each IC, calculate the maximum condition number over all frames
            # and choose the IC that has the smallest maximum.
            for c in range(ii, jj+1):
                Gconds = []
                diag = 0
                for b in range(ii, jj+1):
                    if Gcond_matrix[c, b] == 0.0:
                        Gcond = ICs[c].G_condition(M.xyzs[b].flatten()*ang2bohr)
                        Gcond_matrix[c, b] = Gcond
                    else:
                        Gcond = Gcond_matrix[c, b]
                    Gconds.append(Gcond)
                    diag = max(diag, diag_matrix[c, b])
                Gcond_maxs.append(max(Gconds))
                diags.append(diag)
    
            valid_frames = np.array([k <= max(1, min(diags)) for k in diags])
            sorted_valid_frames = np.argsort(Gcond_maxs)[np.where(valid_frames[np.argsort(Gcond_maxs)])]
            candidate_frames = []
            candidate_ICs = []
    
            def add_candidate_IC(c):
                iic = ii + c
                if ICs[iic] not in candidate_ICs:
                    candidate_frames.append(iic)
                    candidate_ICs.append(ICs[iic])
                    print("(%i, %i): adding the IC from frame %i; diagnostic = %i, Gcond_max = %.5e" % (ii, jj, iic, diags[c], Gcond_maxs[c]))
                
            max_candidates = 5
            if ii == 0:# and diags[0] <= 1:
                add_candidate_IC(0)
                max_candidates += 1
            if jj == n_frames-1:# and diags[jj-ii] <= 1:
                add_candidate_IC(jj-ii)
                max_candidates += 1
            for c in sorted_valid_frames:
                add_candidate_IC(c)
                if len(candidate_ICs) == max_candidates: break
    
            segment_to_ICs.append([(i, j) for i, j in zip(candidate_frames, candidate_ICs)])
        
        # For each segment, these are the frame numbers and corresponding IC systems that can be used to interpolate that segment.
        self.segment_to_ICs = segment_to_ICs

    def build_IC_segments(self):
        print("Building internal coordinates and determining segments:")
        assert len(self.M) == self.n_frames
        M = deepcopy(self.M)
        transfers = self.transfers
        n_frames = self.n_frames
        splice_length = self.splice_length

        segments = []
        ICs = []
        xyzs = [M.xyzs[i].flatten()*ang2bohr for i in range(len(M))]
    
        # Build the list of ICs; these are TRIC coordinates, isolated atoms disconnected, with atom transfers
        for i in range(len(M)):
            M_i = M[i]
            IC = DelocalizedInternalCoordinates(M_i, build=True, connect=False, addcart=False, transfers=transfers, connect_isolated=False)
            IC.build_dlc(xyzs[i])
            ICs.append(IC)

        diag_matrix = np.zeros((len(M), len(M)), dtype=int)
        diag_messages = {}
        for i in range(len(M)):
            for j in range(len(M)):
                diag, messages = ICs[i].diagnostics(xyzs[j], print_thre=2)
                diag_matrix[i, j] += diag
                diag_messages[(i, j)] = messages

        print_map(diag_matrix, "IC diagnostic map", colorscheme=0)
        segments = find_path(diag_matrix)
        segments_filled, segments_spliced = fill_splice_segments(segments, splice_length)
        print("The frame numbers of the spliced segments are:")
        print(segments_spliced)

        # Assign some class variables to be used by other methods
        self.ICs = ICs
        self.diag_matrix = diag_matrix
        self.segments_filled = segments_filled
        self.segments_spliced = segments_spliced
        self.assign_ICs_to_segments()
        self.segment_endpoints = get_segment_endpoints(M, segments_spliced)

    def splice_iterations(self):
        segment_endpoints_orig = deepcopy(self.segment_endpoints)
        segment_endpoints = deepcopy(self.segment_endpoints)
        # M_in contains the initial guess coordinates, which were created by self.initial_guess()
        # from the (optionally prealigned) endpoints.
        M_in = deepcopy(self.M)
        M = deepcopy(self.M)
        M_reac, M_prod = self.endmols
        xyz0, xyz1 = self.endxyzs
        splice_length = self.splice_length
        segments_spliced = self.segments_spliced
        segment_to_ICs = self.segment_to_ICs
        n_frames = self.n_frames
    
        n_cycles = 1000
        min_max_mismatch = 1e10
        last_max_mismatch = 1e10
        increase_count = 0
        decrease_count = 0
        damping = 1.0
        backwards = [[False for c in range(len(segment_to_ICs[a]))] for a in range(len(segments_spliced))]
        clash_pairs = []
        for cycle in range(n_cycles):
            coord_segments = []
            endpt_errs = []
            M.xyzs = []
            M.comms = []
            frame_to_segment = {}
            for a in range(len(segments_spliced)):
                i, j = segments_spliced[a]
                xyzi, xyzj = segment_endpoints[a]
                nDiv = j-i
                success = False
                for c, (iic, IC) in enumerate(segment_to_ICs[a]):
                    if self.verbose: 
                        print("In splice_iterations: c = %i, iic = %i, IC = %s" % (c, iic, IC.__repr__()))
                        if self.verbose >= 2:
                            vali = IC.Prims.calculate(xyzi)
                            valj = IC.Prims.calculate(xyzj)
                            primDiff = IC.Prims.calcDiff(xyzj, xyzi, sync=1)
                            for iPrim in range(len(IC.Prims.Internals)):
                                print("%3i %25s % 9.5f % 9.5f % 9.5f" % (iPrim, IC.Prims.Internals[iPrim], vali[iPrim], valj[iPrim], primDiff[iPrim]))
                    attempt = 0
                    coord_segment, endpt_err, success = interpolate_segment(IC, xyzi, xyzj, nDiv, backward=backwards[a][c], rebuild_dlc=False, verbose=self.verbose)
                    if self.verbose: 
                        print("forward direction: endpt_err = %8.3f success = %i" % (endpt_err, success))
                        if self.verbose >= 2:
                            write_coord_segment(M, coord_segment, "cycle%i_segment%i_IC%i_attempt%i.xyz" % (cycle, a, c, attempt))
                    if success: break
                    attempt = 1
                    coord_segment, endpt_err, success = interpolate_segment(IC, xyzi, xyzj, nDiv, backward=not backwards[a][c], rebuild_dlc=False, verbose=self.verbose)
                    if self.verbose: 
                        print("backward direction: endpt_err = %8.3f success = %i" % (endpt_err, success))
                        if self.verbose >= 2:
                            write_coord_segment(M, coord_segment, "cycle%i_segment%i_IC%i_attempt%i.xyz" % (cycle, a, c, attempt))
                    if success: 
                        backwards[a][c] = not backwards[a][c]
                        break
                if success:
                    # Reorder the segment_to_ICs so the successful one is at the front.
                    reordered_segment_to_ICs = []
                    for cc, (iic, IC) in enumerate(segment_to_ICs[a]):
                        if c == cc:
                            reordered_segment_to_ICs.append((iic, IC))
                    for cc, (iic, IC) in enumerate(segment_to_ICs[a]):
                        if c != cc:
                            reordered_segment_to_ICs.append((iic, IC))
                    segment_to_ICs[a] = reordered_segment_to_ICs
                else:
                    print("Failed to interpolate segment %i-%i" % (i, j))
                    return 1, a
                
                print("Frames %2i-%2i error in final interpolated vs. product structure: %.3e (candidate %i attempt %i)" % (i, j, endpt_err, c, attempt))
                coord_segments.append(np.array(coord_segment).copy())
                endpt_errs.append(endpt_err)
                if a == 0:
                    keep_start = 0
                else:
                    keep_start = splice_length//2
                if a == len(segments_spliced)-1:
                    keep_end = len(coord_segment)
                else:
                    keep_end = len(coord_segment) - splice_length//2
                for k in range(keep_start, keep_end):
                    frame_to_segment[i+k] = a
                M.xyzs += [coord_segment[k].reshape(-1, 3)*bohr2ang for k in range(keep_start, keep_end)]
                M.comms += ['Frame %i-%i interpolated frame %i' % (i, j, i+k) for k in range(keep_start, keep_end)]
    
    
            _, max_mismatch = splice_segment_endpoints(coord_segments, segment_endpoints, segments_spliced, splice_length, damping, verbose=True)
        
            if max_mismatch < min_max_mismatch:
                min_max_mismatch = max_mismatch
                print(">> Current best result (mismatch=%.3e) saved to interpolated_splice.xyz" % max_mismatch)
                include_alignment = False
                if include_alignment:
                    M_append = M_reac[:-1] + M + M_prod[1:]
                    if len(M_append) != len(M):
                        M_append.align()
                        M = EqualSpacing(M_append, frames=len(M), RMSD=False, spline=True)
                else:
                    if len(M_reac) > 1:
                        M = M_reac[0] + M
                    if len(M_prod) > 1:
                        M = M + M_prod[-1]
                    M.align()
                M.write("interpolated_splice.xyz")
    
            if max_mismatch < last_max_mismatch:
                increase_count = 0
                decrease_count += 1
                if decrease_count >= 3 and damping < 1.0:
                    print(">> Mismatch decreasing; resetting damping")
                    damping = 1.0
            else:
                increase_count += 1
                decrease_count = 0
                print(">> Mismatch fails to decrease; increasing damping")
                damping *= 0.8
            last_max_mismatch = max_mismatch
                
            segment_endpoints, _ = splice_segment_endpoints(coord_segments, segment_endpoints, segments_spliced, splice_length, damping, verbose=False)
        
            if max_mismatch > 0.3: continue

            # Check for clashes.
            coord_traj = [M.xyzs[k].flatten()*ang2bohr for k in range(len(M))]
            clash_pairs, clash_new = self.detect_clash_trajectory(coord_traj, clash_known=clash_pairs, altdists=self.min_enddists, verbose=True)
            if clash_new:
                print("Current list of clashing pairs:", ', '.join(["%s%i-%s%i" % (M.elem[i], i+1, M.elem[j], j+1) for i, j in clash_pairs]))
                for a in range(len(segments_spliced)):
                    print(">> Rebuilding ICs for segment %i" % a)
                    rebuilt_segment_to_ICs = []
                    for c, (iic, IC) in enumerate(segment_to_ICs[a]):
                        new_repulsions = sorted(list(set(IC.Prims.repulsions).union(set(clash_pairs))))
                        IC1 = DelocalizedInternalCoordinates(M_in[iic], build=True, connect=False, addcart=False,
                                                             repulsions=new_repulsions, transfers=IC.Prims.transfers, connect_isolated=False)
                        IC1.build_dlc(M_in.xyzs[iic].flatten()*ang2bohr)
                        rebuilt_segment_to_ICs.append((iic, IC1))
                    segment_to_ICs[a] = rebuilt_segment_to_ICs
                    print("Resetting due to clashes")
                    segment_endpoints = deepcopy(segment_endpoints_orig)
                    min_max_mismatch = 1e10
                    last_max_mismatch = 1e10
            else:
                if max_mismatch < 1.8e-3:
                    print("Converged!")
                    return 0, 0
                else:
                    continue
        if cycle == n_cycles - 1:
            print("Not converged after %i cycles; best max-mismatch is %.5f" % (n_cycles, min_max_mismatch))

    def split_IC_segments(self, fail_segment):
        self.segments_filled, self.segments_spliced = split_segments(self.segments_spliced, fail_segment, self.splice_length)
        self.assign_ICs_to_segments()
        self.segment_endpoints = get_segment_endpoints(self.M, self.segments_spliced)
    
    def run_workflow(self):
        if self.do_prealign:
            self.prealign_fragments()
        if self.do_init:
            self.initial_guess()
        self.build_IC_segments()
        status, fail_segment = self.splice_iterations()
        count = 0
        while status != 0:
            self.split_IC_segments(fail_segment)
            status, fail_segment = self.splice_iterations()
            if count == 3:
                print("Failed after 3 splits; exiting")

def main():
    args = parse_interpolate_args(sys.argv[1:])
    params = InterpParams(**args)
    M0 = Molecule(args['input'])
    #interpolator = Interpolator(M0, use_midframes=True, n_frames=0, align_system=True, do_prealign=False)
    interpolator = Interpolator(M0, n_frames= params.nframes, use_midframes=params.optimize, align_system=params.align, do_prealign=params.prealign, verbose=params.verbose)
    interpolator.run_workflow()
    
if __name__ == "__main__":
    main()
