import psi4
import numpy as np
import sys
import os

from ase.io import read,write

from alframework.moleculedata import Molecule

# Used to build string input into psi4 python interface
def build_input_string(X,S,C,M):
    string = "\n"
    string += str(C)+" "+str(M) +"\n"
    for s,x in zip(S,X):
        string += s+" "+"{0:.6f}".format(x[0])+" "+"{0:.6f}".format(x[1])+" "+"{0:.6f}".format(x[2])+"\n"
    string+="\n"
    return string

# PSI4 QM Interface Class
class psi4Generator:

    # Constructor
    def __init__(self,lot,output_store_path='',num_threads=1,memory='500 MB',reference='rhf',scratch_path=None, cc_maxiter=600):
        self.lot = lot
        self.num_threads = num_threads
        self.memory = memory
        self.reference = reference
        self.output_store_path = output_store_path
        self.scratch_path = scratch_path
        self.psi4_io = psi4.core.IOManager.shared_object()
        self.cc_maxiter = cc_maxiter

    def optimize(self, mol, store_file, output_file='output.opt'):

        # Set the scratch path for PSI4
        if self.scratch_path is not None:
            self.psi4_io.set_default_path(self.scratch_path)

        psi4.core.set_output_file(output_file, False)
        psi4.set_num_threads(self.num_threads )

        properties = {}
        properties['species'] = mol.S
        properties['charge'] = mol.Q
        properties['multip'] = mol.M

        mol = psi4.geometry(build_input_string(mol.X, mol.S, mol.Q, mol.M))
        psi4.optimize(self.lot, molecule=mol)
        Xn = np.array(mol.geometry())
        properties['coordinates'] = Xn
        np.save(store_file, Xn)

    def single_point(self, molecule, force_calculation=False, output_file='output.opt'):
        optfile = "psi4-"+molecule.ids+".out"

        # Set the scratch path for PSI4
        if self.scratch_path is not None:
            self.psi4_io.set_default_path(self.scratch_path)

        psi4.core.set_output_file(self.output_store_path+optfile, False)
        psi4.set_memory(self.memory)
        psi4.set_num_threads(self.num_threads)

        psi4.set_options({'reference': self.reference})
        psi4.set_module_options('scf', {'e_convergence': 1e-8,'d_convergence': 1e-8,'MAXITER': 500})
        psi4.set_module_options('CCENERGY', {'MAXITER': self.cc_maxiter})
        properties = {}
        
        # Compute the energy (units: hartree)
        try:
            psi4_mol = psi4.geometry(build_input_string(molecule.X, molecule.S, molecule.Q, molecule.M))
            properties['energy'] = psi4.energy(self.lot, molecule=psi4_mol, properties=["DIPOLE"])

            if force_calculation:
                properties['forces'] = -np.array(psi4.gradient(self.lot))/0.529177249

            molecule = Molecule(0.529177249*np.array(psi4_mol.geometry()),molecule.S,molecule.Q,molecule.M,molecule.ids)
            print('PSI4 COMPLETE:',"psi4-"+molecule.ids+".out")
            return molecule, properties
        except:
            e = sys.exc_info()[0]
            print('-------ERROR-------',"psi4-"+molecule.ids+".out")
            print(len(molecule.S))
            print('PSI4 Interface Error: '+str(e))
            for s,x in zip(molecule.S,molecule.X):
                print(s+' '+str(x[0])+' '+str(x[1])+' '+str(x[2]))
            molecule = Molecule(molecule.X,molecule.S,molecule.Q,molecule.M,molecule.ids,failed=True)
            return molecule, properties
