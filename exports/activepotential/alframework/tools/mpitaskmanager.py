from mpi4py import MPI
import numpy as np
import time

####### Tasking base class ########
# Tasking base class. All tasks used
# within the MPITaskSync class below
# should use this base class to ensure
# all required variables and defaults
# are in place.
###################################
class TaskBase():
    def __init__(self, type=None, index=None, terminate=False, incomplete=False, empty=False):
        self.type = type
        self.index = index
        self.terminate = terminate
        self.incomplete = incomplete
        self.empty = empty

    def getID(self):
        return self.index

    def term(self):
        return self.terminate

    def success(self):
        return not self.incomplete


########## QM task class ##########
# Quantum mechanics calculation tasking class
class QMTask(TaskBase):
    def __init__(self, molecule=None, qmdata=None, index=None, terminate=False, incomplete=False):
        TaskBase.__init__(self, 'QMTASK',index, terminate, incomplete)
        self.molecule = molecule
        self.qmdata = qmdata


######### ML task class ###########
# Machine learning model tasking class
class MLTask(TaskBase):
    def __init__(self, params=None, status=None, stats=None, index=None, terminate=False, incomplete=False):
        TaskBase.__init__(self, 'MLTASK', index, terminate, incomplete)
        self.params = params
        self.status = status
        self.stats  = stats

########## SM task class ##########
# Sampler tasking class
class SMTask(TaskBase):
    def __init__(self, new_mols=[], sample_setup=None, model_details=None, index=None, terminate=False, incomplete=False):
        TaskBase.__init__(self, 'SMTASK', index, terminate, incomplete)
        self.sample_setup = sample_setup
        self.model_details = model_details
        self.new_mols = new_mols

######## Task Synchronizer ##########
# Constructor arguments:
# 1) communicator -- MPI Communicator
# 2) rankid -- MPI rank of this thread
# 3) rank_list -- list of al MPI ranks
#                 in this communicator
#
# Description: This class manages the 
# MPI ranks and tasks, ensuring that 
# all non-master MPI ranks (referred
# to as workers) have work to do.
#
# In this setup, the lowest rank ID of 
# of the communicator is deemed
# 'master'. The master rank spends its 
# time looking for idle task ranks
# and keeping them busy. Master can
# be used externally to gather new
# tasks.
#####################################
class MPITaskSync():
    # Constructor
    def __init__(self, communicator, rank_list):
        self.comm = communicator # MPI communicator
        self.rank = self.comm.Get_rank()
        self.rank_list = sorted(rank_list) # Sorted rank list for self.comm
        self.master_rank = 0 # Master rank ID -- always lowest rank
        self.master = True if self.rank == self.master_rank else False # Am I master?
        self.shutdown = False # Init shutdown to false
        self.numwkers = len(self.rank_list) - 1
        self.idlwkers = self.numwkers
        self.init_workers() # Initial communication with worker threads

        self.task_queue = []

    # Initialize workers
    def init_workers(self):
        if self.master:
            #print('MASTER -- INIT WORKERS!!!',self.rank)
            self.requests = [(self.comm.irecv(source=worker, tag=11), worker) for worker in self.rank_list[1:]]
        self.comm.Barrier()

    # Send termination signal to workers
    # Desc: Master goes into a termination loop, which it will not
    # exit until all worker ranks have been sent a termination signal
    def terminate_workers(self, requests):
        completed_tasks = []
        while len(requests) > 0:
            ready_workers = np.where([req[0].Test() for req in requests])[0]
            if ready_workers.size > 0:
                for worker in ready_workers:
                    returned_task = self.comm.recv(source=self.requests[worker][1])

                    if not returned_task.empty:
                        completed_tasks.append(returned_task)

                    self.comm.send(TaskBase(terminate=True), dest=requests[worker][1])
                for i in ready_workers[::-1]:
                    requests.pop(i)
        return completed_tasks

    # Synchronize with workers
    # Desc: Master enters, looking for any waiting worker ranks
    # If any worker ranks are idle, send them new tasks.
    def __sync_tasks(self, tasks):
        #print('SYNC_TASKS CHECK IN', self.rank, self.master)
        if self.master:
            #print('SYNC_TASKS FOR MASTER')
            completed_tasks = []

            # Shutdown all workers if requested
            if self.shutdown:
                completed_tasks = self.terminate_workers(self.requests)
                return completed_tasks

            # If master has tasks to send to workers then send them
            for rid, req in enumerate(self.requests):
                if len(tasks) > 0:
                    if req[0].Test():
                        returned_task = self.comm.recv(source=req[1])
                        if not returned_task.empty:
                            print("RETURNED TASK:", returned_task, returned_task.success(), isinstance(returned_task, TaskBase))

                        if not returned_task.empty:
                            completed_tasks.append(returned_task)

                        self.comm.send(tasks.pop(-1), dest=req[1])
                        self.requests[rid] = (self.comm.irecv(source=req[1], tag=11), req[1])
            return completed_tasks
        else:
            # Send ready message to master then wait
            req = self.comm.isend(True, dest=self.master_rank, tag=11)
            req.wait()

            #f = open('sync-checktask-'+str(self.rank).zfill(2), 'w')
            #f.write(str(tasks)+'\n')
            #f.close()

            # Send completed tasks to master
            self.comm.send(tasks, dest=self.master_rank)

            # Receive new tasks from master
            recv_task = self.comm.recv(source=self.master_rank)
            self.shutdown = recv_task.term()

            return recv_task

    #def idle_workers(self):


    ### Add new tasks
    def add_to_task_queue(self, new_tasks):
        self.task_queue.extend([task for task in new_tasks])

    ### Data Generation Task Managing Function
    def sync_tasks(self, tasking):
        if self.master:
            Nempty = self.numwkers - len(self.task_queue)
            if Nempty > 0:
                idle_tasks = [TaskBase(empty=True) for i in range(Nempty)]
                #self.task_queue.extend(idle_tasks)
                idle_tasks.extend(self.task_queue)
                self.task_queue = idle_tasks

            new_tasks = self.__sync_tasks(self.task_queue)

            pop_list= []
            for i, task in enumerate(self.task_queue):
                if task.empty:
                    pop_list.append(i)

            if Nempty > 0:
                self.idlwkers = Nempty - len(pop_list)
            else:
                self.idlwkers = 0

            self.task_queue = [task for i, task in enumerate(self.task_queue) if i not in pop_list]

            return new_tasks
        else:
            new_tasks = self.__sync_tasks(TaskBase(incomplete=True) if tasking is None else tasking)
            if not self.term():
                # Do task on worker rank
                return new_tasks

    # Check if shutdown has been requested
    # Returns the bool self.shutdown
    def term(self):
        return self.shutdown