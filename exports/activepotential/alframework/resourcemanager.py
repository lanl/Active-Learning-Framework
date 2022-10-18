from mpi4py import MPI
import numpy as np

from alframework.tools import QMTask, MLTask, MPITaskSync

class CPUMPIResourceManager:

    # Need 2 threads, one will manage training/sampling and one data generation

    # Build a rank map?

    # num_models -- total ML models to train per node
    # num_nodes -- total nodes in computation
    # num_cores -- number of cores per node
    def __init__(self, num_nodes, num_cores):
        self.num_nodes  = num_nodes
        self.num_cores  = num_cores

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.local_rank = self.rank % int(num_cores)

        self.is_master = (self.rank == 0)
        self.cpu_rank_list = [i for i in range(num_cores * num_nodes)]

        # Setup ranks for GPU or CPU usage
        # This sets up two MPI communicators, one for CPUs and one for GPUs
        if self.cpurank():
            self.cpucomm = MPI.COMM_WORLD.Split(1, self.rank)

        # Wait for CPU and GPU comms to build
        self.comm.Barrier()

        if self.cpurank():
            #print('CPU RANK',self.rank,'CPULIST',self.cpu_rank_list)
            self.cputask = MPITaskSync(self.cpucomm, [i for i in range(len(self.cpu_rank_list))])

        # Wait for CPU and GPU comms to build
        self.comm.Barrier()

        if self.rank == 0:
            rank_seeds = np.random.randint(0, 2 ** 32, self.comm.Get_size())
            for r in range(1, rank_seeds.size):
                self.comm.Send(rank_seeds, dest=r, tag=13)
            self.rank_seed = rank_seeds[self.rank]
        else:
            rank_seeds = np.empty(self.comm.Get_size(), dtype=np.int)
            self.comm.Recv(rank_seeds, source=0, tag=13)
            self.rank_seed = rank_seeds[self.rank]

        # Seed this generator with the rank seed
        np.random.seed(self.rank_seed)

    def get_local_random_seed(self):
        return np.random.randint(0, 2 ** 32)

    # Return true if this is a CPU rank
    def cpurank(self):
        return self.rank in self.cpu_rank_list

    def call_out_rank_cpu(self):
        self.cpucomm.Barrier()

    def terminated(self):
        return self.cputask.term()


class MPIResourceManager:

    # Need 2 threads, one will manage CPU work and one will manage GPU work

    # Build a rank map?

    # num_models -- total ML models to train per node
    # num_gpus -- total gpus available per node
    # num_nodes -- total nodes in computation
    # num_cores -- total number of cores in the computation
    def __init__(self, num_gpus, num_nodes, num_cores, max_hardware_gpus=None):

        if max_hardware_gpus is not None:
            self.max_hw_gpus = max_hardware_gpus
        else:
            self.max_hw_gpus = num_gpus

        self.num_gpus   = num_gpus
        self.num_nodes  = num_nodes
        self.num_cores  = num_cores

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.local_rank = self.rank % int(num_cores+num_gpus)

        self.is_master = (self.rank == 0)
        self.cpu_rank_list = [i + n * (self.num_gpus + self.num_cores) for n in range(self.num_nodes) for i in range(self.num_cores) if i + n * (self.num_gpus + self.num_cores) != 0]
        self.gpu_rank_list = [i + n * (self.num_gpus + self.num_cores) + self.num_cores for n in range(self.num_nodes) for i in range(self.num_gpus)]

        self.cpu_rank_list.insert(0, 0)
        self.gpu_rank_list.insert(0, 0)

        # Setup ranks for GPU or CPU usage
        # This sets up two MPI communicators, one for CPUs and one for GPUs
        if self.cpurank():
            self.cpucomm = self.comm.Split(1, self.rank)

        # A weird problem with split is blocking when not all ranks go through a split.
        # This hack allows me to create the cpu and gpu comms I need which both contain rank 0
        if self.rank in [i for i in np.arange(self.size) if i not in self.cpu_rank_list]:
            self.hack_comm_1 = self.comm.Split(2, self.rank)

        # GPU communicator
        if self.gpurank():
            self.gpucomm = self.comm.Split(3, self.rank)

        # A weird problem with split is blocking when not all ranks go through a split.
        # This hack allows me to create the cpu and gpu comms I need which both contain rank 0
        if self.rank in [i for i in np.arange(self.size) if i not in self.gpu_rank_list]:
            self.hack_comm_2 = self.comm.Split(4, self.rank)

        # Wait for CPU and GPU comms to build
        #print('CPULIST:', self.cpu_rank_list)
        #print('GPULIST:', self.gpu_rank_list)

        #self.comm.Barrier()
        #exit(0)
        self.comm.Barrier()

        if self.cpurank():
            #print('CPU RANK',self.rank,'CPULIST',self.cpu_rank_list)
            self.cputask = MPITaskSync(self.cpucomm, [i for i in range(len(self.cpu_rank_list))])

        if self.gpurank(): 
            #print('GPU RANK',self.rank,'GPULIST',self.gpu_rank_list)
            self.gputask = MPITaskSync(self.gpucomm, [i for i in range(len(self.gpu_rank_list))])
            self.gpuid = self.gpucomm.Get_rank() % self.max_hw_gpus

        # Wait for Task Syncs
        self.comm.Barrier()

        if self.rank == 0:
            rank_seeds = np.random.randint(0, 2**32, self.comm.Get_size())
            for r in range(1, rank_seeds.size):
                self.comm.Send(rank_seeds, dest=r, tag=13)
            self.rank_seed = rank_seeds[self.rank]
        else:
            rank_seeds = np.empty(self.comm.Get_size(), dtype=np.int)
            self.comm.Recv(rank_seeds, source=0, tag=13)
            self.rank_seed = rank_seeds[self.rank]

        print('SEED CHECK:',self.rank,self.rank_seed)

        # Seed this generator with the rank seed
        np.random.seed(self.rank_seed)

    def get_local_random_seed(self):
        return np.random.randint(0, 2 ** 32)
        
    # Return true if this is a CPU rank
    def cpurank(self):
        return self.rank in self.cpu_rank_list

    # Return true if this is a GPU controlling rank
    def gpurank(self):
        return self.rank in self.gpu_rank_list

    # Return true if this is a GPU controlling worker rank
    def gpuidx(self):
        if self.gpuworker():
            return

    def call_out_rank_gpu(self):
        if self.gpurank():
            self.gpucomm.Barrier()

    def call_out_rank_cpu(self):
        if self.cpurank():
            self.cpucomm.Barrier()
        
    def terminated(self):
        if self.cpurank():
            return self.cputask.term()
        else:
            return self.gputask.term()
