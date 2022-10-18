from mpi4py import MPI
import os
import sys
import time
import alframework as Alf
from alframework.moleculedata import Molecule
from alframework.tools import QMTask, MLTask, SMTask
import numpy as np
import json

Nnodes = 1 # Total number of nodes. Needs to be the same as in slurm submission file
Ngpus  = 8  # Number of GPUs per node
Ncores = 12 # Number of cores per node

cpus = 5

# Working directory
root = os.getcwd() + "/"
workdir = "data_path/"
h5store = "h5store/"
ckstore = "ckstore/"
scratch = root+"scratch/"

try:
    os.mkdir("dynamics")
    os.mkdir("dynamics/crds")
except:
    print("dynamics exists")

# Training Parameters JSON
tpjson = "ani_training.json"
mdlpath = "models/"
Nmodels = 8

# Sampling Params JSON
spjson = "mlmdsampler.json"

# Sampling setup
setup = {}
setup['size'] = 1
setup['cell_range'] = [11.0, 14.0]
setup['min_dist'] = 1.5
setup['max_patience'] = 500


#np.random.seed(1)

rm = Alf.MPIResourceManager(Ngpus, Nnodes, Ncores)

if rm.is_master:
    if not os.path.isdir(root + workdir):
        os.mkdir(root + workdir)

    if not os.path.isdir(root + workdir + h5store):
        os.mkdir(root + workdir + h5store)

    if not os.path.isdir(root + workdir + ckstore):
        os.mkdir(root + workdir + ckstore)

    if not os.path.isdir(root + workdir + mdlpath):
        os.mkdir(root + workdir + mdlpath)

rm.comm.Barrier()

if rm.cpurank():
    # Define psi4 QM interface and calculation settings
    qm = Alf.qminterfaces.psi4_interface.psi4Generator("wb97x-D/cc-pvtz",root + workdir + ckstore, num_threads=cpus, memory='40000 MB', scratch_path='/mnt/local/nvme/scratch/')

    # QM data generator ranks are always looking for new data
    dg = Alf.datageneration.qmgenerator(rm.is_master, qm, root + workdir + h5store, forces=True, store_prefix='new_data_it-')


if rm.gpurank():
    # Define ML model interface
    ml = Alf.mlinterfaces.NeuroChemTrainer(Nmodels, rm.gpuid, periodic=False)

    # Define ML ensemble trainer
    et = Alf.mltraining.MLEnsembleTrainer(rm.is_master, ml, tpjson, root + workdir + h5store, root + workdir + mdlpath,
                                          Nmodels, seed=rm.get_local_random_seed())

    # Define sampling method
    mds = Alf.samplers.NormalMDSampler(rank_id=rm.rank, sample_molecules_path=root+"xyz/", sample_json=root+spjson, path_to_dynamics=root+'dynamics/', seed=rm.get_local_random_seed())

    # Define Sample generator
    sm = Alf.samplegeneration.smgenerator(rm.is_master, mds, rm.gpuid,
                                          ase_calculator_function=Alf.mlinterfaces.NeuroChemCalculator_SigMax)

rm.comm.Barrier()
if rm.is_master:
    for i in range(1):
        rm.gputask.add_to_task_queue(et.get_ensemble_train_task())


threshold = 16 # number of MD simulations per AL cycle

it = 0
cpu_tasking = None
gpu_tasking = None
start_time = time.time()

current_models = 0
gen_systems = []

# if new geometries are added to seeding xyz's during AL cycles, some oldest are deleted if total number of seeding xyz's > max_num_xyz + 10. Only max_num_xyz newest are kept.
# Do not make a difference in current settings.
max_num_xyz=1500000000
samples_running = 0
failed_data = 0
old_failed_data = 0
working_ensemble_1 = []


sys.stdout = open('output-'+str(rm.rank)+'.out', 'w')

print('RANK INFORMATION:',rm.rank,MPI.Get_processor_name())

file = open('print.out', 'w+')
file.write(os.getcwd())
file.write('\n')
# Iterate until termination is called for
while not rm.terminated():

    # Delete some oldest seeding xyz's
    xyz_files = os.listdir(root+'xyz')
    if len(xyz_files) > max_num_xyz + 10:
        full_path = [root+"xyz/{0}".format(x) for x in xyz_files]
        for i in range(0,len(xyz_files) - max_num_xyz):
            #xyz_files = os.listdir('xyz')
            file.flush()
            try:
                oldest_file = min(full_path, key=os.path.getctime)
                file.write(oldest_file+'\n')
                os.remove(oldest_file)
                full_path.remove(oldest_file)
            except:
                continue
                
    # Master determines what to add to cpu/gpu work queues
    if rm.is_master:

        # Time termination checking
        if (time.time() - start_time) > 40*60*60: # AL max time
            print('!!!!TERMINATING!!!!')
            rm.cputask.shutdown = True
            rm.gputask.shutdown = True

        if len(dg.new_data) >= threshold:
            print('Training new models and storing new data...')
            if et.train_new_ensemble():
                dg.store_current_data()
                samples_running = 0

                for i in range(1):
                    rm.gputask.add_to_task_queue(et.get_ensemble_train_task())

        # Add new data to the task queue for QM data generation
        if len(sm.new_data) > 0:
            print('ADDING NEW QM CALCULATIONS...')
            rm.cputask.add_to_task_queue([QMTask(mol) for mol in sm.new_data])
            sm.new_data = []

        if old_failed_data < dg.count_failed:
            failed_data = dg.count_failed - old_failed_data
            samples_running -= failed_data
            old_failed_data = dg.count_failed

        # If new models add sampling
        working_ensemble = et.get_new_ensembles()
        if len(working_ensemble) > 0:
            working_ensemble_1 = working_ensemble

        for ensemble in working_ensemble_1:
            #print(working_ensemble)
            Nsamp = threshold - samples_running
            samples_running += Nsamp
            print('Adding new sampling tasks for ensemble:', ensemble, ' Number: ', Nsamp)
            #sm.sm.ens_path = ensemble
            rm.gputask.add_to_task_queue([SMTask(sample_setup=setup, model_details={'model_path': ensemble, 'Nn': Nmodels}) for i in range(Nsamp)])

        # Print status
        if it % 10 == 0:
            print('iter:', it,
                  'idlc:', rm.cputask.idlwkers,
                  'idlg:', rm.gputask.idlwkers,
                  'dgnd:', len(dg.new_data),
                  'smnd:', len(sm.new_data),
                  'ensl:', len(et.comp_ensembles),
                  'cpuq:', len(rm.cputask.task_queue),
                  'gpuq:', len(rm.gputask.task_queue),
                  'dgfl:', dg.count_failed,
                  'mdfl:', et.count_failed,
                  )

    # CPU work
    if rm.cpurank():
        # Sync CPU tasks
        cpu_tasking = rm.cputask.sync_tasks(cpu_tasking)

        # Task out data generation work
        cpu_tasking = dg.data_generation_tasking(cpu_tasking)

    # GPU work
    if rm.gpurank():
        # Sync CPU tasks
        gpu_tasking = rm.gputask.sync_tasks(gpu_tasking)

        # Task out sampling work
        gpu_tasking = sm.sample_tasking(gpu_tasking)

        # Task out training work
        gpu_tasking = et.ml_train_tasking(gpu_tasking)

    sys.stdout.flush()

    time.sleep(0.1)
    it += 1

# Store the new data
#if rm.is_master:
#    dg.store_current_data()

rm.comm.Barrier()
