#!/usr/bin/env python3

import MDAnalysis
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.dihedrals import *
from MDAnalysis.analysis.rms import RMSF
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

parser = argparse.ArgumentParser(description='Analyze a MD trajectory using MDAnalysis')
parser.add_argument("topology", help="Topology PDB file from OpenMM")
parser.add_argument("trajectory", help="Trajectory file")
args = parser.parse_args()

# KEEP all the hardcoded output file names as this is how your script will be graded.
print('Starting MD analysis...')

# preprocess trajectory
u_rmsd = MDAnalysis.Universe(args.topology, args.trajectory)
protein_rmsd = u_rmsd.select_atoms('protein')

# align the trajectory to the first frame (removing overall rotation/translation)
align.AlignTraj(u_rmsd, u_rmsd, select='backbone', in_memory=True).run()

# RMSD calculations
print('Computing RMSD...')
rmsd_first_analysis = rms.RMSD(u_rmsd, u_rmsd, select='backbone', ref_frame=0)
rmsd_first_analysis.run()

# the third column of the rmsd array contains RMSD values
rmsd_first = rmsd_first_analysis.rmsd[:, 2]

# compute RMSD relative to last frame
last_frame = u_rmsd.trajectory.n_frames - 1
rmsd_last_analysis = rms.RMSD(u_rmsd, u_rmsd, select='backbone', ref_frame=last_frame)
rmsd_last_analysis.run()
rmsd_last = rmsd_last_analysis.rmsd[:, 2]

np.save('rmsd_first.npy', rmsd_first)
np.save('rmsd_last.npy', rmsd_last)

# plot RMSD vs. frame
plt.figure()
plt.plot(rmsd_first, label='first')
plt.plot(rmsd_last, label='last')
plt.xlabel('Frame')
plt.ylabel('RMSD (Å)')
plt.title('RMSD Analysis')
plt.legend()
plt.savefig('rmsd-sim.png')
plt.close()

# RMSF Calculations
# compute RMSF for every protein atom (for writing to pdb file)
u_rmsf = MDAnalysis.Universe(args.topology, args.trajectory)
protein_rmsf = u_rmsf.select_atoms('protein')

# assign computed RMSF to the atom B-factor
from MDAnalysis.analysis.align import AverageStructure, AlignTraj
avg_struct = AverageStructure(u_rmsf, select='backbone')
avg_struct.run()

# create a reference universe using the topology, assign its backbone positions to the average
ref = MDAnalysis.Universe(args.topology)
ref_backbone = ref.select_atoms('backbone')
ref_backbone.positions = avg_struct.results['positions']

# align u_rmsf to the average structure using backbone atoms
AlignTraj(u_rmsf, ref, select='backbone', in_memory=True).run()

# compute RMSF for the alpha carbons (CA)
print('Computing RMSF for alpha carbons...')
calphas = protein_rmsf.select_atoms('name CA')
rmsf_ca_analysis = RMSF(calphas).run()
ca_rmsf = rmsf_ca_analysis.rmsf
np.save('ca_rmsf.npy', ca_rmsf)

# plot the RMSF for CA atoms
plt.figure()
plt.plot(calphas.resids, ca_rmsf, linestyle='-', label='RMSF')
plt.xlabel('Residue Number')
plt.ylabel('RMSF (Å)')
plt.title('RMSF (Alpha Carbons)')
plt.legend()
plt.savefig('rmsf-sim.png')
plt.close()

# compute RMSF for all atoms (after alignment to average structure)
print('Computing RMSF for all atoms...')
rmsf_all_analysis = RMSF(protein_rmsf).run()
# set each atom’s tempfactor (B-factor) equal to its RMSF value
protein_rmsf.atoms.tempfactors = rmsf_all_analysis.rmsf
# write out a PDB file with tempfactors set
protein_rmsf.atoms.write('rmsf.pdb')

# run next two analyses on residue 62
r = u_rmsd.select_atoms('protein and resid 62')

# ramachandran analysis
print('Running Ramachandran analysis...')
rama = Ramachandran(r).run()
rama_angles = rama.results['angles']
np.save('ramachandran.npy', rama_angles)
rama.plot(color='black', marker='.', ref=True)
plt.savefig("ramachandran-sim.png")
plt.close()

# janin analysis
print('Running Janin analysis...')
janin = Janin(r).run()
janin_angles = janin.results['angles']
np.save('janin.npy', janin_angles)
janin.plot(color='black', marker='.', ref=True)
plt.savefig('janin-sim.png')
plt.close()

print('MD analysis complete.')