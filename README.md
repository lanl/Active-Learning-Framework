# Active-Learning-Framework (C22072)
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  

## Workflow in brief
- The code trains an ensemble of NN interatomic potentials *data_path/models/* using initial training set *data_path/h5store/initial_training_set_125.h5*.  
- Then MD sampling is performed using NN ensemble interfaced with ASE.  
- Bias potential code: *exports/neurochem-build/lib/ase_interface.py*  
- Bias potential interfaced with ASE-based MD sampler: *exports/activepotential/alframework/samplers/normal_md_sampling.py*  
- New data is stored in *data_path/h5store*.  
- The cycle repeats untill time limit reached or manually stopped.  


## Repository sctructure	

*exports* contains 4 folders:  
- *activepotential* - general AL framework  
- *ANI-Tools* - scripts for training and data processing  
- *boost_1_63_0* -   
- *neurochem-build* - NeuroChem compiled binaries  

*example* contains 5 folders:  
- *350K_udd-al*, *350K_md-al*, *600K_md-al*, and *1000K_md-al* jobs with settings used for the generation of data discussed in the paper.  In each of them:  
  - *xyz* folder contains a set of starting geometries for MD simulations
  - *data_path* contains 3 folders:  
    - *h5store* contains initial training set and sampled data.
    - *models* (created after the job submission) contains ensembles of models from each AL iteration.
    - *ckstore* (created after the job submission) contains QC (psi4) out files.
  - *dymanics* (created after the job submission) contains log files with some information from each MD simulation.  
  - *al_ani_md.py* - main AL script  
  - *ani_training.json* - NN and training parameters  
  - *mlmdsampler.json* - AL and MD parameters  
  - *sbatch_1node.sh* - slurm submission file for a single node (requires configuration)  
  - *sbatch_2nodes.sh* - slurm submission file for multiple nodes (requires configuration)  
- *example_ase_ani* - python script with ANI interfaced with ase
	


## Build environment
- Allocate node, e.g.
`salloc -p atdm-ml`  
- create and activate new env  
`conda create --name udd-al`  
`source activate udd-al`
- install required python packages  
`conda install psi4 psi4-rt python=3.6 mpi4py==3.0.3 pandas scikit-learn==0.23.1 -c psi4`  
`pip install --upgrade --user ase`

##   		 AL execution			 
###    change {full_path} to proper folders

#### Tested with:
	OS: Red Hat Enterprise Linux 8.4 (Ootpa)
	Kernel: Linux 4.18.0-305.25.1.el8_4.x86_64
	Architecture: x86-64
	CPU: Intel Xeon Gold 6138
	GPU: NVIDIA TITAN V
	CUDA: 11.4.2
	gcc (GCC) / GNU Fortran (GCC):  8.4.1 20200928 (Red Hat 8.4.1-1)
	MPICH: 3.3a2

*Typical install time on a "normal" (but GPU-accelerated) desktop computer: ~1hr*  
*Expected run time for demo on a "normal" desktop computer with 1 GPU: ~10min for the first AL iteration (no bias potential, T=350K). Then time increases as more data generated.*

#### Running on single machine

1) in al_ani_md.py
   - set Nnodes to 1
   - Ngpus - number of GPUs on your machine
   - Ncores - available cores on your machine
   - cpus - cpus for psi4 calculations

2) run commands below in terminal or see slurm submission file sbatch_1node.sh
   - add anaconda path, e.g. `export PATH="{full_path}/anaconda3/bin:$PATH"`
   - activate env `source activate udd-al`  
   -  add paths to mpich, AL framework, boost, ANI-tools, NeuroChem  
*change {full_path} to the location of mpich*  
`export PATH={full_path}/mpich-install/bin/:$PATH`  
*change {full_path} to the location of ALF_binary/exports*  
`EXP_PATH="{full_path}/ALF_binary/exports"`  
`export PYTHONPATH="$EXP_PATH/activepotential/:$PYTHONPATH"`  
`export LD_LIBRARY_PATH=$EXP_PATH/boost_1_63_0:$LD_LIBRARY_PATH`  
`export PYTHONPATH="$EXP_PATH/ANI-Tools:$PYTHONPATH"`  
`export LD_LIBRARY_PATH=$EXP_PATH/neurochem-build/lib:$LD_LIBRARY_PATH`  
`export PYTHONPATH=$EXP_PATH/neurochem-build/lib/:$PYTHONPATH`  
`export PATH=$EXP_PATH/neurochem-build/bin/:$PATH`
3) mpirun -n {Ngpus+Ncores} python al_ani_md.py, e.g. `mpirun -n 20 python al_ani_md.py`


#### Running on multiple nodes
(see slurm submission file sbatch_2nodes.sh)

1) in al_ani_md.py
   - set Nnodes to the number of nodes
   - Ngpus - number of GPUs per node
   - Ncores - cores per node
   - cpus - cpus for psi4 calculations

2) in sbatch_2nodes.sh
   - change {#SBATCH -N ???} accordingly
   - the last line should be: mpirun -n {Nnodes*(Ngpus+Ncores)} -ppn {Ngpus+Ncores} python al_ani_md.py
     - e.g. for 2 nodes with Ngpus=8 and Ncores=12: `mpirun -n 40 -ppn 20 python al_ani_md.py`


## AL parameters

#### In mlmdsampler.json

   - temperature (float) - MD temperature in K
   - timestep (float) - MD timestep in fs
   - md_max_steps (integer) - MD steps limit
   - enable_e_b (boolean) - energy uncertainty bias on/off
   - enable_f_b (boolean) - force uncertainty bias on/off
   - functional ("exp") - bias potential shape. Only exponentional supporter 
   - alpha_e/alpha_f (float) - bias potential magnitude (A) in a.u. (energy/force uncertainty)

   - beta_e/beta_f (float) - bias potential width (B) in a.u.  (energy/force uncertainty)

   - ramping_interval_a, rate_e_a/rate_f_a (integer):
     - bias potential magnitudes alpha_e/alpha_f are multiplied by rate_e_a/rate_f_a every ramping_interval_a steps.

   - ramping_interval_b, rate_e_b/rate_f_b (integer):
     - bias potential widths beta_e/beta_f are multiplied by rate_e_b & rate_f_b every ramping_interval_b steps.

   - iteration_to_set_bias (integer) - AL iteration to turn bias on
     - !!! there is no parameter to change temperature at specific AL iteration. Therefore, change temperature in mlmdsampler.json when AL reaches desired iteration.

   - bias_step_min & bias_step_max (integer) - at each AL iteration, the bias is on starting at a random step between bias_step_min & bias_step_max

   - sigma_stop_E (float) - energy uncertainty criterion (stop criterion), eV(!!!)
   - sigma_stop_F (float) - force uncertainty criterion (stop criterion), eV/A(!!!)

   - save_to_restart (boolean) - save final MD geometries to use them as starting structures for subsequent AL iterations
   - save_to_restart_frequency (integer) - saving frequency. 1 - save all. 2 - randomly save 50%. 3 - randomly save 33%. etc.
   - return_xyz_with_highest_ - specifies which MD step to save if MD reached the time step limit and save_to_restart==True.
     - "Fs": step with highest Fsigma
     - "Es": step with highest Esigma
     - "sum": step with highest Fsigma + Esigma
     - anything else: the last step


   - Parameters for testing purposes (do not change!):
     - step_to_start_ramping
     - sigma_stop_sum
     - md_stop_type
     - md_steps
     - alpha_factor
     - sigma_factor
     - alpha
     - bias_atoms


#### In ani_training.json

##### data_setup
   - Ntrain, Nvalid, Ntest - training:validation:test split such that (Ntrain-Nvalid-Ntest):Nvalid:Ntest
   - hold_out - percentage of hold out set (e.g., 0.1 - 10%)

##### aev_params
   - elements - all element types (e.g., ["H","C","N","O"])
   - NRrad - number of radial functions in radial descriptor
   - Rradcut - cutoff radius for radial part
   - NArad - number of radial functions in angular descriptor
   - NAang - number of angular functions in angular descriptor
   - Aradcut - cutoff radius for angular part
   - x0 - position of the first atom-centered function

##### input_params
   - eta - 
   - energy - train to energies
   - force - train to forces
   - fmult -
   - feps - 
   - dipole - 
   - cdweight - 
   - repuls - 
   - tolr - 
   - tbtchsz - training batch size
   - vbtchsz - validation batch size
   - nkde - 
   - pbc - periodic boundary conditions flag
   - bboxx, bboxy, bboxz - box size for pbc

##### layers
   - nodes - number of nodes in a layer
   - activation - 
   - type - 
   - l2norm - l2 regularization
   - l2valu - l2 value
