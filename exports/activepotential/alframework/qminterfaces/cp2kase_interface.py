# Import ASE
import ase
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.cp2k import CP2K
from ase import Atoms

# Import PYTHON stuff
import os
import re
import time
import shutil
import pickle as pkl
import numpy as np

# IMPORT ALF LIBS
from alframework.moleculedata import Molecule

# CP2K QM Interface Class
class CP2KGenerator:

    # Constructor
    def __init__(self,rankid,input_template,cp2k_shell='cp2k_shell.ssmp',scratch='',output_store='',num_threads=1,store_d3=False):
        self.rankid = rankid
        self.input_template = input_template
        self.cp2k_shell = cp2k_shell
        self.num_threads = num_threads
        self.scratch = scratch
        self.output_store = output_store
        self.store_d3 = store_d3
        self.counter = 0

        self.scratch_path = self.scratch+'/worker-'+str(self.rankid).zfill(4)
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        command = 'cd '+self.scratch_path+' && ' + self.cp2k_shell

        self.existing_pkls = np.array([f for f in os.listdir(output_store) if f[-2:] == '.p'])
        #for i in range(10000):
        #    print('Existing pkls:',self.existing_pkls)

        if not os.path.isdir(self.scratch_path):
            os.mkdir(self.scratch_path)
        else:
            shutil.rmtree(self.scratch_path)
            os.mkdir(self.scratch_path)

        with open(self.input_template, 'r') as myfile:
            inp = myfile.read()
        inp = re.sub(r'PROJECT_NAME\s+[^\s]+', '', inp, count=1)

        self.settings = dict(command = command,
                        auto_write = False,
                        basis_set = None,
                        basis_set_file = None,
                        charge = None,
                        cutoff = None,
                        force_eval_method = None,
                        inp = inp,
                        max_scf = None,
                        potential_file = None,
                        pseudo_potential = None,
                        stress_tensor = None,
                        uks = None,
                        poisson_solver = None,
                        xc = None,
                        print_level = 'LOW',
                        debug=False)

        #self.calc = CP2K(**settings)

    def __del__ (self):
        shutil.rmtree(self.scratch_path)
        
    def optimize(self):
        print('')

    def single_point(self, molecule, force_calculation=False, output_file='output.opt'):
        self.calc = CP2K(**self.settings)

        # Compute the energy (output units: eV and Angstroms -> Hartree and Angstrom)
        try:
            if self.existing_pkls.size > 0:
                compute = np.where(self.existing_pkls == 'data-'+molecule.ids+'.p')[0].size == 0
            else: 
                compute = True

            if compute:
                self.calc.label = 'data-'+str(self.rankid).zfill(4)+'-'+str(self.counter).zfill(4)

                if molecule.periodic:
                    atoms = Atoms(molecule.S,positions=molecule.X,cell=molecule.C,pbc=(True,True,True))
                else:
                    atoms = Atoms(molecule.S,positions=molecule.X)

                properties = {}

                atoms.set_calculator(self.calc)
                properties['energy'] = (1.0/units.Hartree)*atoms.get_potential_energy()

                if force_calculation:
                    properties['forces'] = (1.0/units.Hartree)*atoms.get_forces()

                try:
                    self.calc.__del__()
                except:
                    pass

                if molecule.periodic:
                    molecule = Molecule(np.array(atoms.get_positions()),molecule.S,molecule.Q,molecule.M,molecule.ids,C=molecule.C)
                else:
                    molecule = Molecule(np.array(atoms.get_positions()),molecule.S,molecule.Q,molecule.M,molecule.ids)

                print("SCF CHECK:",self.calc.label,":",molecule.ids,"SCF run NOT converged" in open(self.scratch_path+"/"+self.calc.label+'.out','r').read())
                output_data = open(self.scratch_path+"/"+self.calc.label+'.out','r').read()

                if self.store_d3:
                    d3file = [f for f in os.listdir(self.scratch_path) if ".dftd" in f][0]
                    d3data = open(self.scratch_path+"/"+d3file,'r').read()

                if "SCF run NOT converged" not in output_data:
                    #shutil.move(self.scratch_path+"/"+self.calc.label+'.out', self.output_store+"/"+self.calc.label+'.out')
                    output_file = open(self.output_store+"/data-"+molecule.ids+'.out',"w")
                    output_file.write(output_data)
                    output_file.close()
                    
                    if self.store_d3:
                        output_file = open(self.output_store+"/data-"+molecule.ids+'.dftd',"w")
                        output_file.write(d3data)
                        output_file.close()
                
                    for f in os.listdir(self.scratch_path):
                        os.remove(self.scratch_path+'/'+f)

                    pkl.dump( {"molec":molecule,"props":properties}, open( self.output_store+"/data-"+molecule.ids+'.p', "wb" ) )

                    self.counter += 1
                    return molecule, properties
                else:
                    #shutil.move(self.scratch_path+"/"+self.calc.label+'.out', self.output_store+"/"+self.calc.label+'.out')
                    output_file = open(self.output_store+"/data-"+molecule.ids+'-FAIL.out',"w")
                    output_file.write(output_data)
                    output_file.close()

                    for f in os.listdir(self.scratch_path):
                        os.remove(self.scratch_path+'/'+f)

                    self.counter += 1
                    return Molecule(np.array(atoms.get_positions()),molecule.S,molecule.Q,molecule.M,molecule.ids,failed=True), properties
            else:
                loaded_data = pkl.load( open( self.output_store+"/data-"+molecule.ids+'.p', "rb" ) )
                print("LOADED FROM FILE:",self.calc.label,":",molecule.ids)
                self.counter += 1
                return loaded_data["molec"], loaded_data["props"]
        except:
            shutil.move(self.scratch_path+"/"+self.calc.label+'.out', self.output_store+"/"+self.calc.label+'.out')
            for f in os.listdir(self.scratch_path):
               os.remove(self.scratch_path+'/'+f)

            print('!!FAILED CALC!!')
            return Molecule(np.array(atoms.get_positions()),molecule.S,molecule.Q,molecule.M,molecule.ids,failed=True), properties



