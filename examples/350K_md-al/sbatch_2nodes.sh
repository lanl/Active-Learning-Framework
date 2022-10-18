#!/bin/sh
#SBATCH --job-name="MD-AL" # number of nodes  
#SBATCH -N 2 # number of nodes 
#SBATCH --mem-per-cpu=6400 # memory pool per thread 
#SBATCH -t 7-00:00 # time (D-HH:MM) 
#SBATCH -o slurm.job.%j.out # STDOUT 
#SBATCH -e slurm.job.%j.err # STDERR 
#SBATCH -p atdm-ml
#SBATCH --qos=unlimited

#####################################################
####    change {full_path} to proper folders	 ####
#####################################################

###### ANACONDA Python #######
export PATH="{full_path}/anaconda3/bin:$PATH"

# path to env
source activate udd-al

###### mpich ######
export PATH={full_path}/mpich-install/bin/:$PATH

# change to full path of ALF_binary/exports
EXP_PATH="{full_path}/ALF_binary/exports"

export PYTHONPATH="$EXP_PATH/activepotential/:$PYTHONPATH"

#------------------------ BOOST EXPORTS-------------------------
export LD_LIBRARY_PATH=$EXP_PATH/boost_1_63_0:$LD_LIBRARY_PATH

export PYTHONPATH="$EXP_PATH/ANI-Tools:$PYTHONPATH"

#---------------------- NeuroChem EXPORTS-------------------------
export LD_LIBRARY_PATH=$EXP_PATH/neurochem-build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$EXP_PATH/neurochem-build/lib/:$PYTHONPATH
export PATH=$EXP_PATH/neurochem-build/bin/:$PATH

echo "Launching python script..."
mpirun -n 40 -ppn 20 python al_ani_md.py


