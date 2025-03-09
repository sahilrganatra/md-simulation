#!/usr/bin/env python3

from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm.unit import picosecond, femtoseconds, nanometer
from pdbfixer.pdbfixer import PDBFixer
import sys, argparse

parser = argparse.ArgumentParser(description='Simulate a PDB using OpenMM')
parser.add_argument("--pdb", required=True, help="PDB file name")
parser.add_argument("--temperature", type=float,default="300", help="Temperature for simulation in Kelvin")
parser.add_argument("--steps", type=int, default=125000000, help="Number of 2fs time steps")
parser.add_argument("--etrajectory", type=str,default="etrajectory.dcd", help="Equilibration  dcd trajectory name")
parser.add_argument("--trajectory", type=str,default="trajectory.dcd", help="Production dcd trajectory name")
parser.add_argument("--einfo", type=argparse.FileType('wt'), default=sys.stdout, help="Equilibration simulation info file")
parser.add_argument("--info", type=argparse.FileType('wt'), default=sys.stdout, help="Production simulation info file")
parser.add_argument("--system_pdb", type=str, default="system.pdb", help="PDB of system, can be used as topology")

args = parser.parse_args()

# load PDB and add any missing residues/atoms/hydrogens (at pH 7) (pdbfixer)
fixer = PDBFixer(filename=args.pdb)
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)

# remove MG (not handled by the forcefield)
fixer.removeHeterogens(keepWater=False)

modeller = Modeller(fixer.topology, fixer.positions)

# pre-solvation minimization
print('Starting pre-solvation minimization...')
min_forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
min_system = min_forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
min_integrator = LangevinMiddleIntegrator(args.temperature*kelvin, 1/picosecond, 2*femtoseconds)
min_simulation = Simulation(modeller.topology, min_system, min_integrator)
min_simulation.context.setPositions(modeller.positions)
min_simulation.minimizeEnergy()

# update modeller w/ minimized positions
minimized_positions = min_simulation.context.getState(getPositions=True).getPositions()
modeller = Modeller(modeller.topology, minimized_positions)

# solvate protein w/ octahedral water box
print('Solvating system...')
explicit_ff = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
modeller.addSolvent(explicit_ff, model='tip3p', padding=1.0*nanometer, boxShape='octahedron', neutralize=True)

with open(args.system_pdb, 'w') as f:
    PDBFile.writeFile(modeller.topology, modeller.positions, f)

# rebuild system for simulation (explicit solvent)
print('Setting up full solvated system...')
system = explicit_ff.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0*nanometer,
    constraints=HBonds
)

# add barostat - start w/ freq 0
barostat = MonteCarloBarostat(1*atmosphere, args.temperature*kelvin, 0)
system.addForce(barostat)

# create new integrator for production sim
integrator = LangevinMiddleIntegrator(args.temperature*kelvin, 1/picosecond, 2*femtoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

# minimize the full solvated system
print('Minimizing full solvated system...')
simulation.minimizeEnergy()

# equilibration
print('Starting equilibration...')
stateReporter = StateDataReporter(
    args.einfo, 
    reportInterval=50, 
    step=True, 
    temperature=True, 
    volume=True, 
    potentialEnergy=True, 
    speed=True
)
dcdReporter = DCDReporter(args.etrajectory, 500)
simulation.reporters.append(stateReporter)
simulation.reporters.append(dcdReporter)

# phase 1 NVT warm-up over 50 ps
target_temp = args.temperature
for i in range(100):
    new_temp = (i + 1) * (target_temp / 100)
    integrator.setTemperature(new_temp*kelvin)
    simulation.step(250) # 250 steps * 2 fs = 0.5 ps per iteration

# phase 2 NPT equilibration over 20 ps
barostat.setFrequency(25)
simulation.step(10000) # 10000 steps * 2 fs = 20 ps

simulation.reporters = []
production_interval = int(10*picosecond / (2*femtoseconds))
simulation.reporters.append(StateDataReporter(
    args.info,
    reportInterval=production_interval,
    step=True,
    volume=True,
    potentialEnergy=True,
    speed=True
))
simulation.reporters.append(DCDReporter(args.trajectory, production_interval))

print('Starting production simulation...')
simulation.step(args.steps)
print('Simulation complete.')