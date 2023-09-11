#!/usr/bin/env python

from openmm.unit import *
from openmm import *
from openmm.app import *

pdb = PDBFile('ethane.pdb')

ff = ForceField('ethane.xml')

system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None)

integrator = VerletIntegrator(0.001*picosecond)

simulation = Simulation(pdb.topology, system, integrator)

simulation.context.setPositions(pdb.positions)

simulation.saveState('state.xml')

sysxml = XmlSerializer.serialize(system)
with open("system.xml", "w") as f:
    f.write(sysxml)

# print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
