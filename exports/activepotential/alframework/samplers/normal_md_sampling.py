'''
#####################################
#### JSON PARAMETERS DESCRIPTION ####
#####################################
o temperature
	md temperature in K
	Values: any positive float

o timestep:
	md timestep
	Values: any positive float

o functional
	Type of bias function.
	Values: “log”, “log_k”, “parabola”, “exp”
		log: -alpha*log(beta*x + c)
		log_k: -alpha*log(x/beta + c)
		parabola: alpha*(beta*x – c)^2
		exp: alpha*exp(-beta*x)

o alpha
	values: "auto" - automated selection; "manual" - manual selection (below)

o alpha_factor
	For automated selection only. Multiplies the selected alpha.

o alpha_f and alpha_e
	alpha values of force and energy bias functions.
	Values: any float 

o beta_f and beta_e
	beta values of force and energy bias functions.
	Values: any float

o rate_e_a and rate_f_a
	alpha_e and alpha_f are multiplied by these factors every specified interval (ramping_interval_a)
	Values: any float

o rate_e_b and rate_f_b
	beta_e and beta_f are multiplied by these factors every specified interval (ramping_interval_b)
	Values: any float

o bias_step_min and bias_step_max
	The number of initial non-bias md steps is a random number between "bias_step_min" and "bias_step_max". If you don’t need the bias at all, set "bias_step_min" higher than "md_max_steps" rather than just setting alpha to 0 because bias increases the computational time. Always set "bias_step_min" larger than "bias_step_max". 
	Values: positive integers: bias_step_min < bias_step_max

o md_steps
	Defines how often storeenergy function is called. It’s better to keep it 1.
	Values: any positive integer

o md_max_steps
	Maximum number of md steps.
	Values: any positive integer

o md_stop_type
	Defines the md stop criteria.
	Values:
		“E” – stop md when Esig is higher than  automatically determined value
		“F” – stop md when Fsig is higher than  automatically determined value
		“Sum” – stop md when Fsig+Esig is higher than automatically determined value
        "Separate" - stop md when Esigma or Fsigma are higher than automatically determined value
		“Separate_manual” – stop md when either Esigma is higher than "sigma_stop_E" or Fsigma is higher than "sigma_stop_F"
        "Separate_manual_std" - stop md when either E_stddev is higher than "sigma_stop_E" or F_stddev is higher than "sigma_stop_F"

o sigma_factor
	For automated sigma selection only. Multiplies the selected sigma.

o sigma_stop_sum, sigma_stop_F, sigma_stop_E
	See md_stop_type.
	Values: any positive float

o step_to_start_ramping
	Number of md steps before parameters (alpha and beta) ramping is on.
	Values: positive integer

o ramping_interval_a and ramping_interval_b
	"alpha_f"/"alpha_e" are multiplied by "rate_f_a"/"rate_e_a" every "ramping_interval_a" steps. "beta_f"/"beta_e" are multiplied by "rate_f_b"/"rate_e_b" every "ramping_interval_b steps". For example, the first incrementation of "alpha_f" will be at [step_to_start_ramping + ramping_interval_a] step.
	Values: positive integer

o return_xyz_with_highest_
	Defines which step is returned.
	Values:
		"Fs": step with highest Fsigma
		"Es": step with highest Esigma
		"sum": step with highest Fsigma + Esigma
		anything else: the last step

o save_to_restart
	If set to “yes”, some final (or with highest Fsig, Esig, sum) md steps from random runs wil be saved to xyz folder for further restart. That is, they will be taken (probably) as starting geometries in next md runs.
	Values: “yes” or anything else

o save_to_restart_frequency
	1 out of "save_to_restart_frequency"  md runs will save the final (or with highest Fsig, Esig, sum) step to xyz.
	Values: any positive integer
############
############
############
'''

import os
import numpy as np
import sys
from alframework.moleculedata import Molecule
from alframework.tools import random_rotation_matrix
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write
from ase import units
from ase import neighborlist
from ase import Atoms
from ase.optimize import LBFGS
import time
import json
from ase_interface import ANIENS
from ase_interface import ensemblemolecule
from ase_mc_npt import MCBarostat
import pyNeuroChem as pync
import  ase
import time
import os
os.environ["OMP_NUM_THREADS"] = "4"
import  ase
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase.optimize.fire import FIRE as QuasiNewton
from ase.md.nvtberendsen import NVTBerendsen
from ase.md import MDLogger
from ase.io import read, write
from random import randint
from ase.parallel import world
class NormalMDSampler:

    def __init__(self, rank_id, sample_molecules_path, sample_json, path_to_dynamics, seed = -1,update_systems=True):
        self.molsamplepath = sample_molecules_path
        self.ini_id = rank_id
        self.samp_id = 0
        self.ens_path = path_to_dynamics
        self.sample_json = sample_json

        self.update_systems = update_systems
        if not update_systems:
            self.molecule_files = os.listdir(self.molsamplepath)

        if seed != -1:
            np.random.seed(seed)

    def sample(self, setup, ml=None):

        #list xyz structures and choose random one
        if self.update_systems:
            molecule_files = os.listdir(self.molsamplepath)
            mol_filename = molecule_files[np.random.randint(0, len(molecule_files)-1)]
            mol = read(self.molsamplepath+mol_filename, parallel = False)
        else:
            molecule_files = self.molecule_files
            mol_filename = molecule_files[np.random.randint(0, len(molecule_files)-1)]
            mol = read(self.molsamplepath+mol_filename, parallel = False)

        if ml is not None:

            dir = self.ens_path

            #mlmdsampler.json loader
            train_params = json.load(open(self.sample_json))

            #T = float(str(sys.argv[1])) # Temperature
            T = train_params["bias_params"]["temperature"]

            # Dynamics file
            xyzfile1 = dir + 'crds/mdcrd_NPT_'+ ml['model_path'][-5:-1] +"-"+ str(int(T))+"-" + str(self.ini_id).zfill(5)+"-"+str(self.samp_id).zfill(5)+'.xyz'

            # Trajectory file
            trajfile1 = dir + 'traj_NPT_' + ml['model_path'][-5:-1] +"-"+ str(int(T))+"-" + str(self.ini_id)+"-"+str(self.samp_id)+'.dat'

            dt = train_params["bias_params"]["timestep"]

            #setcell = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
            #print('Cell Info:',setcell,np.prod(np.sum(setcell,axis=0)))
            #mol.set_cell((setcell))
            #mol.set_pbc((True, True, True))

            # Set ANI calculator
            mol.set_calculator(ml['model'])

            #np.random.seed(0)

            dyn = Langevin(mol, dt * units.fs, T * units.kB, 0.015, communicator=None)

            # Open MD output
            mdcrd1 = open( xyzfile1, 'w')
            #traj1  = open(trajfile1, 'w')

            # load force bias parameters
            # load starting alpha
            #self.weight_f=train_params["bias_params"]["alpha_f"]
            # load starting beta
            self.beta_f = 1 / (2 * (train_params["bias_params"]["beta_f"]*(2*np.sqrt(len(mol))))**2 )
            self.beta_e = 1 / (2 * (train_params["bias_params"]["beta_e"]*(2*np.sqrt(len(mol))))**2 )  ##1 / ( 2 * (beta_e*(2*np.sqrt(10))/27.21138505)**2 )

            if train_params["bias_params"]["alpha"] == "auto":
                self.weight_f = train_params["bias_params"]["alpha_factor"]*(self.beta_f*ml["uqmax"]["fuqmax"]*ml["uqmax"]["fuqmax"] + ml["uqmax"]["fuqmax"])/self.beta_f
                self.weight_e = train_params["bias_params"]["alpha_factor"]*(self.beta_e*ml["uqmax"]["euqmax"]*ml["uqmax"]["euqmax"] + ml["uqmax"]["euqmax"])/self.beta_e
            elif train_params["bias_params"]["alpha"] == "manual":
                self.weight_f = train_params["bias_params"]["alpha_f"]
                self.weight_e = train_params["bias_params"]["alpha_e"]
            else:
                self.weight_f = train_params["bias_params"]["alpha_factor"]*(self.beta_f*ml["uqmax"]["fuqmax"]*ml["uqmax"]["fuqmax"] + ml["uqmax"]["fuqmax"])/self.beta_f
                self.weight_e = train_params["bias_params"]["alpha_factor"]*(self.beta_e*ml["uqmax"]["euqmax"]*ml["uqmax"]["euqmax"] + ml["uqmax"]["euqmax"])/self.beta_e
                

            #load energy bias parameters
            #self.weight_e=train_params["bias_params"]["alpha_e"]



            # create file for some md information. stored in dynamics folder
            self.unc_f=open(dir + ml['model_path'][-5:-1] +"-"+ str(self.ini_id).zfill(5)+"-"+str(self.samp_id).zfill(5)+"unc_f.dat", "w+")
            self.unc_f.write('Filename: '+mol_filename+'\n')
            self.unc_f.write('ML Model:'+ml['model_path']+'\n')
            self.unc_f.write('esigmax/fsigmax: '+"{:.5f}".format(ml["uqmax"]["euqmax"])+'/'+"{:.5f}".format(ml["uqmax"]["fuqmax"])+'\n')
            #self.unc_f.write("Step stddev     Fstddev   alpha_e  alpha_f   beta_e        beta_f      Highest_achieved_Es  Highest_achieved_Fs\n")

            # these uncertainties are used in md while loop to check if the criteria met
            self.E_variance=0
            self.F_variance=0
            self.E_variance_std=0
            self.F_variance_std=0

            # used in md while loop to check if the maximum step achieved
            self.step_iteration=0

            # used in alpha and beta ramping up to check the interval
            self.counter_to_iterate_a=1
            self.counter_to_iterate_b=1

            # set the random number of initial non-bias steps (between bias_step_min and bias_step_max)
            self.bias_step = np.random.randint(train_params["bias_params"]["bias_step_min"], train_params["bias_params"]["bias_step_max"])
            self.unc_f.write("bias step: " + str(self.bias_step) + "\n")
            self.unc_f.write("Alpha: " + train_params["bias_params"]["alpha"] + "\n")
            self.unc_f.write("Alpha_E ramping increment: " + str(train_params["bias_params"]["rate_e_a"]) + "\n")
            self.unc_f.write("Alpha_F ramping increment: " + str(train_params["bias_params"]["rate_f_a"]) + "\n")
            self.unc_f.write("Ramping interval (a): " +str(train_params["bias_params"]["ramping_interval_a"]) + "\n")
            self.unc_f.write("Alpha factor: " +str(train_params["bias_params"]["alpha_factor"]) + "\n")
            self.unc_f.write("Sigma factor: " +str(train_params["bias_params"]["sigma_factor"]) + "\n")
            self.unc_f.write("Enable energy bias: " +str(train_params["bias_params"]["enable_e_b"]) + "\n")
            self.unc_f.write("Enable force bias: " +str(train_params["bias_params"]["enable_f_b"]) + "\n")
            self.unc_f.write("Biased atoms: " +str(train_params["bias_params"]["bias_atoms"]) + "\n")

            self.unc_f.write("Step stddev       Fstddev      alpha_e   alpha_f  beta_e     beta_f\n")

            # used to print highest achieved sigma at every step
            self.Fs_max = 0
            self.Es_max = 0
            self.Fs_Es_sum_max = 0

            # Define the printer and bias functions
            def storeenergy(a=mol, d=dyn, b=mdcrd1, md_output=False, ml=ml):  # store a reference to atoms in the definition.
                """Function to print the potential, kinetic and total energy."""
                epot = a.get_potential_energy() / len(a)
                ekin = a.get_kinetic_energy() / len(a)

                stddev =  a.calc.nc.intermediates['E_sig']
                Fstddev = a.calc.nc.intermediates['F_sig']

                #cell = a.get_cell()
                #cell = np.array([cell[0,0],cell[1,1],cell[2,2]])

                # store xyz of the highest uncertainty. in case Fsig criteria won't be met, this xyz is returned.
                if Fstddev > self.Fs_max:
                    self.Fs_max = Fstddev
                    self.Fs_max_old = a.calc.Fstddev
                    self.coords_to_return_fs = a.get_positions()

                # the same for Esig criteria. not implemented yet
                if stddev > self.Es_max:
                    self.Es_max = stddev
                    self.Es_max_old = a.calc.Estddev
                    self.coords_to_return_es = a.get_positions()

                if stddev + Fstddev > self.Fs_Es_sum_max:
                    self.Fs_Es_sum_max  = stddev + Fstddev
                    self.coords_to_return_sum = a.get_positions()


                self.E_variance = stddev
                self.F_variance = Fstddev
                self.F_variance_std = a.calc.Fstddev
                self.E_variance_std = a.calc.Estddev
                #print('from norm_md:  ',self.E_variance_std)

                self.step_iteration = d.get_number_of_steps()

                # start ramping up alpha nad beta if md step > "step_to_start_ramping"
                if d.get_number_of_steps() > train_params["bias_params"]["step_to_start_ramping"]:
                    self.counter_to_iterate_a += 1

                    #multiply alpha by "rate_e" every specified interval (json)
                    if self.counter_to_iterate_a > train_params["bias_params"]["ramping_interval_a"]:
                        self.counter_to_iterate_a = 1
                        self.weight_e *= train_params["bias_params"]["rate_e_a"]
                        self.weight_f *= train_params["bias_params"]["rate_f_a"]

                if d.get_number_of_steps() > train_params["bias_params"]["step_to_start_ramping"]:
                    self.counter_to_iterate_b += 1
                    if self.counter_to_iterate_b > train_params["bias_params"]["ramping_interval_b"]:
                        self.counter_to_iterate_b = 1
                        self.beta_e *= train_params["bias_params"]["rate_e_b"]
                        self.beta_f *= train_params["bias_params"]["rate_f_b"]

                # set bias if specified md step achieved
                if d.get_number_of_steps() > self.bias_step and int(ml['model_path'][-5:-1]) >= train_params["bias_params"]["iteration_to_set_bias"]:
           
                    # Bias - Exponent: a*exp(-b*x)
                    def bias_Efunc_ex(sigma_f, sigma_e):
                        return self.weight_f * np.exp(-self.beta_f * sigma_f) + self.weight_e * np.exp(-self.beta_e * sigma_e) 

                    def bias_Ffunc_ex(sigma_f, dsigma, sigma_e, sigma_grad_e):
                        return self.weight_f * self.beta_f * np.exp(-self.beta_f * sigma_f) * dsigma + \
				self.weight_e * self.beta_e * np.exp(-self.beta_e * sigma_e) * sigma_grad_e


                    # Set the bias potential
                    if train_params["bias_params"]["functional"] == "parabola":
                        mol.calc.set_sigmabias(bias_Efunc_p, bias_Ffunc_p, enable_energy_bias=train_params["bias_params"]["enable_e_b"],
						enable_force_bias=train_params["bias_params"]["enable_f_b"], bias_atoms=train_params["bias_params"]["bias_atoms"], epsilon=0.001, disable_ani=False)
                    elif train_params["bias_params"]["functional"] == "exp":
                        mol.calc.set_sigmabias(bias_Efunc_ex, bias_Ffunc_ex, enable_energy_bias=train_params["bias_params"]["enable_e_b"],
						enable_force_bias=train_params["bias_params"]["enable_f_b"], bias_atoms=train_params["bias_params"]["bias_atoms"], epsilon=0.001, disable_ani=False)
                    elif train_params["bias_params"]["functional"] == "log":
                        mol.calc.set_sigmabias(bias_Efunc_log, bias_Ffunc_log, enable_energy_bias=train_params["bias_params"]["enable_e_b"],
						enable_force_bias=train_params["bias_params"]["enable_f_b"], bias_atoms=train_params["bias_params"]["bias_atoms"], epsilon=0.001, disable_ani=False)
                    elif train_params["bias_params"]["functional"] == "log_e":
                        mol.calc.set_sigmabias(bias_Efunc_log_e, bias_Ffunc_log_e, epsilon=0.001, disable_ani=False)
                    elif train_params["bias_params"]["functional"] == "log_k":
                        mol.calc.set_sigmabias(bias_Efunc_log_k, bias_Ffunc_log_k, enable_energy_bias=train_params["bias_params"]["enable_e_b"],
						enable_force_bias=train_params["bias_params"]["enable_f_b"], bias_atoms=train_params["bias_params"]["bias_atoms"], epsilon=0.001, disable_ani=False)
                    else:
                        # not sure where it is printed out
                        print("no bias")

                # print out some information in dynamics folder
                self.unc_f.write(str(self.step_iteration)+"    "+str(round(self.E_variance_std,5))+ "      " +str(round(self.F_variance_std,5))+"      "+str(round(self.weight_e,5))+"       "+str(round(self.weight_f,5))+"      "+ "\n")

                if md_output:
                    b.write(str(len(a)) + '\n' + str(ekin / (1.5 * units.kB)) + ' Step: ' + str(d.get_number_of_steps()) + '\n')
                    c = a.get_positions(wrap=False)
                    for j, i in zip(a, c):
                        b.write(str(j.symbol) + ' ' + str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n')

                print('Step: %d Energy per atom: Epot = %.6f  Ekin = %.3f (T=%.3fK)  '
                      'Etot = %.6f Esig = %.2f Fsig = %.2f' % (d.get_number_of_steps(), epot, ekin, ekin / (1.5 * units.kB), epot + ekin, stddev, Fstddev))

            print('Starting equil run...')

            # Set the printer
            dyn.attach(storeenergy, interval=1)

            s_time = time.time()

            # choose stop criteria: E - energy uncertainty, F - force uncertainty, Sum - Fs+Es, Separate - either Es or Fs
            if train_params["bias_params"]["md_stop_type"] == "E":
                while self.E_variance < ml["uqmax"]["euqmax"]*train_params["bias_params"]["sigma_factor"] and self.step_iteration < train_params["bias_params"]["md_max_steps"]:
                    dyn.run(train_params["bias_params"]["md_steps"] )

            if train_params["bias_params"]["md_stop_type"] == "F":
                while self.F_variance < ml["uqmax"]["fuqmax"]*train_params["bias_params"]["sigma_factor"] and self.step_iteration < train_params["bias_params"]["md_max_steps"]:
                    dyn.run(train_params["bias_params"]["md_steps"] )

            if train_params["bias_params"]["md_stop_type"] == "Sum":
                while self.E_variance + self.F_variance < (ml["uqmax"]["euqmax"]+ml["uqmax"]["fuqmax"])*train_params["bias_params"]["sigma_factor"] and self.step_iteration < train_params["bias_params"]["md_max_steps"]:
                    dyn.run(train_params["bias_params"]["md_steps"] )

            if train_params["bias_params"]["md_stop_type"] == "Separate":
                while self.E_variance < ml["uqmax"]["euqmax"]*train_params["bias_params"]["sigma_factor"] and self.F_variance < ml["uqmax"]["fuqmax"]*train_params["bias_params"]["sigma_factor"]:
                    if self.step_iteration >= train_params["bias_params"]["md_max_steps"]:
                        break
                    dyn.run(train_params["bias_params"]["md_steps"] )


            if train_params["bias_params"]["md_stop_type"] == "Separate_manual":
                while self.E_variance < train_params["bias_params"]["sigma_stop_E"] and self.F_variance < train_params["bias_params"]["sigma_stop_F"]:
                    if self.step_iteration >= train_params["bias_params"]["md_max_steps"]:
                        break
                    dyn.run(train_params["bias_params"]["md_steps"] )

            if train_params["bias_params"]["md_stop_type"] == "Separate_manual_std":
                while self.E_variance_std < train_params["bias_params"]["sigma_stop_E"] and self.F_variance_std < train_params["bias_params"]["sigma_stop_F"]:
                    if self.step_iteration >= train_params["bias_params"]["md_max_steps"]:
                        break
                    dyn.run(train_params["bias_params"]["md_steps"] )
            
            print('Total time:',time.time()-s_time)

            mdcrd1.close()

            #aens.cleanup()
        # set unique index
        samp_number = 'molecule-'+str(self.ini_id).zfill(5)+'-'+str(self.samp_id).zfill(5)
        self.unc_f.write("Highest Fsigma: " + str(round(self.Fs_max, 4)) + "\n")
        self.unc_f.write("Highest Esigma: " + str(round(self.Es_max, 4)) + "\n")
        self.unc_f.write("Highest Fsigma + Esigma: " + str(round(self.Fs_Es_sum_max, 4)) + "\n")
        self.unc_f.write("Number of MD steps: " + str(self.step_iteration) + "\n")

        # return xyz with highest Fs if criteria not met
        if train_params["bias_params"]["return_xyz_with_highest_"] == "Fs" and self.step_iteration >= train_params["bias_params"]["md_max_steps"]:
            mol.set_positions(self.coords_to_return_fs)
            self.unc_f.write("Returned: highest Fsigma\n")

        elif train_params["bias_params"]["return_xyz_with_highest_"] == "Es" and self.step_iteration >= train_params["bias_params"]["md_max_steps"]:
            mol.set_positions(self.coords_to_return_es)
            self.unc_f.write("Returned: highest Esigma\n")

        elif train_params["bias_params"]["return_xyz_with_highest_"] == "sum" and self.step_iteration >= train_params["bias_params"]["md_max_steps"]:
            mol.set_positions(self.coords_to_return_sum)
            self.unc_f.write("Returned: highest Fsigma + Esigma\n")

        else:
            self.unc_f.write("Returned: last md step\n")

        self.unc_f.write("Total time: " + str(time.time()-s_time) + "\n")

        # randomly save 1 out of "save_to_restart_frequency" xyz for later restart (json)
        if train_params["bias_params"]["save_to_restart"] and self.step_iteration < train_params["bias_params"]["md_max_steps"]:
            if np.random.randint(0, train_params["bias_params"]["save_to_restart_frequency"]) == 0:
                fi = open(self.molsamplepath + ml['model_path'][-5:-1] +"-"+samp_number + ".xyz", "w+")
                to_save=np.concatenate((np.array([mol.get_chemical_symbols()]).T, mol.get_positions()), axis=1)
                fi.write(str(len(to_save)) + "\n")
                fi.write("chno\n")
                np.savetxt(fi, to_save, fmt='%s')
                fi.close()

        self.samp_id+=1
        self.unc_f.close()
        return list([Molecule(X=mol.get_positions(),S=mol.get_chemical_symbols(),Q=0,M=1,ids=samp_number)])

