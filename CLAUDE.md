# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

geomeTRIC is a geometry optimization library for molecular structures. It computes optimized molecular geometries by calling external quantum chemistry or molecular mechanics software for energy and gradient calculations.

## Build and Install Commands

```bash
# Install from source
pip install .

# Install in development mode
pip install -e .

# Install from PyPI
pip install geometric
```

## Testing

```bash
# Run all tests
pytest -v geometric/tests/

# Run tests with coverage
pytest -v --cov=geometric geometric/tests/

# Run a single test file
pytest -v geometric/tests/test_molecule.py

# Run a specific test function
pytest -v geometric/tests/test_molecule.py::test_function_name
```

Tests require optional dependencies depending on which engines are being tested. Many tests are skipped if the corresponding QM/MM software is not installed. See `geometric/tests/addons.py` for skip markers like `using_psi4`, `using_openmm`, `using_gaussian`, etc.

## CLI Entry Points

- `geometric-optimize` - Main geometry optimizer (from `geometric.optimize:main`)
- `geometric-neb` - Nudged elastic band calculations (from `geometric.neb:main`)
- `run-ase` - ASE engine runner (from `geometric.ase_engine:main`)

## Architecture

### Core Modules

- **optimize.py** - Main optimization driver. Contains `Optimizer` class that manages the optimization loop, trust radius, and convergence checking. Entry point is `Optimize()` function.

- **engine.py** - Engine base class and implementations for each supported software package. Each engine (TeraChem, QChem, Psi4, Gaussian, OpenMM, Gromacs, etc.) implements `calc_new()` to compute energy and gradient at given coordinates.

- **molecule.py** - `Molecule` class for reading/writing molecular structures in many file formats (.xyz, .pdb, .mol2, etc.). Handles coordinate transformations and trajectory storage.

- **internal.py** - Internal coordinate systems. Key classes:
  - `CartesianCoordinates` - Simple Cartesian coordinates
  - `PrimitiveInternalCoordinates` - Bonds, angles, dihedrals
  - `DelocalizedInternalCoordinates` - TRIC (Translation-Rotation Internal Coordinates), the default and recommended system

- **prepare.py** - `get_molecule_engine()` function that parses input files and returns Molecule + Engine objects. Also handles constraint parsing.

- **params.py** - `OptParams` class containing all optimization parameters (convergence criteria, trust radius, TS/IRC settings).

- **step.py** - Step algorithms including trust radius updates, Hessian updates, and root-finding (Brent's method).

- **neb.py** - Nudged Elastic Band implementation for finding reaction paths.

- **nifty.py** - Utility functions, unit conversions (bohr2ang, ang2bohr, kcal2au, etc.), and logging setup.

### Data Flow

1. User provides input file (e.g., `.tcin`, `.psi4in`, `.qcin`) and coordinates (`.xyz`)
2. `prepare.get_molecule_engine()` creates Molecule and Engine objects
3. `Optimizer` builds internal coordinates and iteratively:
   - Calls `engine.calc_new()` for energy/gradient
   - Transforms gradient to internal coordinates
   - Computes step using trust radius method
   - Checks convergence criteria

### Supported Engines

QM: TeraChem, Q-Chem, Psi4, Gaussian (09/16), Molpro, CFOUR, QUICK
MM: OpenMM, Gromacs
Other: ASE, QCEngine API

## Unit Conventions

- Coordinates: Angstroms in Molecule, Bohr internally during optimization
- Energies: Hartrees (atomic units)
- Gradients: Hartrees/Bohr

## Versioning

Uses versioneer for automatic versioning from git tags. Version format follows PEP 440.
