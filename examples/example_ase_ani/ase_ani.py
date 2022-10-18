from ase.io import read, write
import ase_interface
from ase_interface import ANIENS
from ase_interface import ensemblemolecule

# Molecule file
molfile = 'glyc.xyz'

# Load molecule
mol = read(molfile)

# Load model. Set the path to ensemble
ntdir = '{path_to_ensemble}/data_path/models/ensemble-0001/'
cns = ntdir + 'train0/rHCNO-4.6R_32-3.5A_a8-8.params'
sae = ntdir + 'train0/sae_linfit.dat'
nnf = ntdir + 'train'
Nn=8
aens = ensemblemolecule(cns, sae, nnf, Nn, 0)

# Set ANI calculator
mol.set_calculator(ANIENS(aens))

# Calculate properties
Energy = mol.get_potential_energy()
Forces = mol.get_forces()

print('Energy:\n', Energy)
print('Forces:\n', Forces)
