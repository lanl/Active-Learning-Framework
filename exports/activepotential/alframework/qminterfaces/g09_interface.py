import numpy as np
import os
import re
import subprocess
import random
import shutil

from alframework.moleculedata import Molecule

# G09 QM Interface Class
class g09Generator:

    # Constructor
    def __init__(self,rankid,lot,local_cores=1,local_memory='10GB',scratch_path='scratch/',checkpt_path='checkpt/',imp_solvents=None):
        self.lot = lot
        self.rankid = rankid
        self.datacounter = 0
        self.local_cores=local_cores
        self.local_memory=local_memory
        self.gauchk = 'checkpoint-'
        self.scratch_path = scratch_path
        self.checkpt_path = checkpt_path
        self.imp_solvents = imp_solvents

    def optimize(self, mol, store_file, num_threads=1, output_file='output.opt'):
        print('NOT IMPLEMENTED')

    def write_gaussian_input(self,S,X,Q,M,ids):
        scratch_space_path = self.scratch_path+'scratch-'+str(self.rankid).zfill(4)+'-'+str(self.datacounter).zfill(4)+'/'
        if not os.path.isdir(scratch_space_path):
            os.mkdir(scratch_space_path)

        inputname = scratch_space_path+'input-'+str(self.rankid).zfill(4)+'-'+str(self.datacounter).zfill(4)+'.inp'
        otputname = self.checkpt_path+'/'+'input-'+str(self.rankid).zfill(4)+'-'+str(self.datacounter).zfill(4)+'.log'
        checkname = self.checkpt_path+'/'+'input-'+str(self.rankid).zfill(4)+'-'+str(self.datacounter).zfill(4)+'.chk'

        f = open (inputname,'w')
        f.write('%RWF='+'data.rwf\n')
        f.write('%Int='+'data.int\n')
        #f.write('%INP='+scratch_space_path+'data.inp\n')
        f.write('%Skr='+'data.skr\n')
        f.write('%D2E='+'data.d2e\n')
        f.write('%NoSave\n')
        f.write('%Chk='+checkname+'\n')
        f.write('%NProcShared='+str(self.local_cores)+'\n')
        f.write('%Mem='+self.local_memory+'\n')
        
        if self.imp_solvents:
            self.solvent_key = np.random.choice(list(self.imp_solvents.keys()))
            if self.solvent_key != "Vacuum":
                f.write('# '+self.lot+' force pop=(full,Hirshfeld) SCRF=(SMD,Solvent='+self.solvent_key+')\n')
            else:
                f.write('# '+self.lot+' force pop=(full,Hirshfeld)\n')
        else:
            f.write('# '+self.lot+' force pop=(full,Hirshfeld)\n')

        f.write('\n')
        if self.imp_solvents:
            f.write('Solvent: '+str(self.solvent_key)+' ids:' + str(ids) + '\n')
        else:
            f.write('ids:'+str(ids)+'\n')
        f.write('\n')
        f.write(str(Q)+' '+str(M)+'\n')
        for s,x in zip(S,X):
            f.write(s+' '+"{0:.7f}".format(x[0])+' '+"{0:.7f}".format(x[1])+' '+"{0:.7f}".format(x[2])+'\n')
        f.write('\n')
        f.close()
        return inputname, otputname, checkname       
 
    def run_gaussian_job(self,infile,otfile,ckfile):
        scratch_space_path = self.scratch_path+'scratch-'+str(self.rankid).zfill(4)+'-'+str(self.datacounter).zfill(4)+'/'
        proc = subprocess.Popen("cd "+scratch_space_path+" && g09 < "+ infile + " > " + otfile, shell=True, stdout=subprocess.PIPE)
        outs, errs = proc.communicate()
        #print(outs,errs)

        proc = subprocess.Popen(["formchk", ckfile, ckfile+'.fchk'], stdout=subprocess.PIPE)
        outs, errs = proc.communicate()

        os.remove(ckfile)
        #print(outs,errs)

    def check_normal_termination(self,otfile):
        return 'Normal termination of Gaussian' in open(otfile,'r').read()

    def read_gaussian_chkpoint(self,ckfile):
        file = open(ckfile+'.fchk','r').read()

        re_ele = re.compile('Nuclear charges\s+?R\s+?N=\s+?\d+?\n([\S\s]+?)Current cartesian coordinates')
        re_xyz = re.compile('Current cartesian coordinates\s+?R\s+?N=\s+?\d+?\n([\S\s]+?)Force Field')
        re_enr = re.compile('Total Energy\s+?R\s+?(\S+?)\n')
        re_frc = re.compile('Cartesian Gradient\s+?R\s+?N=\s+?\d+?\n([\S\s]+?)Dipole Moment')
        #re_dip = re.compile('Dipole Moment\s+?R\s+?N=\s+?\d+?\n([\S\s]+?)QEq coupling tensors')
        re_dip = re.compile('Dipole Moment\s+?R\s+?N=\s+?\d+?\n([\S\s]+?)(?=QEq coupling tensors|Quadrupole Moment)')

        ele_map = {'1':'H',
                   '6':'C',
                   '7':'N',
                   '8':'O',
                   '9':'F',
                   '16':'S',
                   '17':'Cl'}

        S = [ele_map[str(e)] for e in np.array(np.array([i for i in "".join(re_ele.findall(file)[0].split('\n')).split(' ') if len(i) > 0],dtype=np.float64),dtype=int)]
        X = np.array([i for i in "".join(re_xyz.findall(file)[0].split('\n')).split(' ') if len(i) > 0],dtype=np.float64).reshape(-1,3)
        E = float(re_enr.findall(file)[0])
        F = -np.array([i for i in "".join(re_frc.findall(file)[0].split('\n')).split(' ') if len(i) > 0],dtype=np.float64).reshape(-1,3)
        D = np.array([i for i in "".join(re_dip.findall(file)[0].split('\n')).split(' ') if len(i) > 0],dtype=np.float64)
        return X,E,F,D,np.zeros(len(S))

    def single_point(self, molecule, force_calculation=False, output_file='output.opt'):
        self.molec = molecule
        self.force = force_calculation
        self.dipol = True
        self.cm5cg = True

        infile,otfile,ckfile = self.write_gaussian_input(molecule.S,molecule.X,molecule.Q,molecule.M,molecule.ids)

        self.run_gaussian_job(infile,otfile,ckfile)

        properties = {}
        if self.check_normal_termination(otfile):
           X,E,F,D,C = self.read_gaussian_chkpoint(ckfile)
           properties['energies'] = E
           properties['forces'] = F/0.529177249
           properties['dipole'] = D*0.529177249
           properties['cm5chg'] = C
 
           if self.imp_solvents:
               properties['solvent'] = self.imp_solvents[self.solvent_key]

           molecule = Molecule(0.529177249*np.array(X),molecule.S,molecule.Q,molecule.M,molecule.ids)
        else:
           molecule = Molecule(molecule.X,molecule.S,molecule.Q,molecule.M,molecule.ids,failed=True)

        # Clean up
        os.remove(infile)
        shutil.rmtree(self.scratch_path+'scratch-'+str(self.rankid).zfill(4)+'-'+str(self.datacounter).zfill(4)+'/')

        self.datacounter += 1
        return molecule, properties
