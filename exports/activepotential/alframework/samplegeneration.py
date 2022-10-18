from mpi4py import MPI

import numpy as np

import alframework as alf
from alframework.moleculedata import Molecule
from alframework.tools import SMTask

#import pyanitools as pyt

import copy
import time

class smgenerator:

    def __init__(self, master, sampler, gpuid=0, ase_calculator_function=None):
        self.master = master
        self.sm = sampler
        self.calc = ase_calculator_function
        self.gpuid = gpuid

        self.new_data = []

        self.terminate = False
        self.task_index = 0

        self.count_failed = 0
        self.count_stored = 0

    def set_calculator_function(self, ase_calculator_function):
        self.calc = ase_calculator_function

    def set_task_index(self):
        taskid = self.task_index
        self.task_index += 1
        return taskid

    ### Sample Generation Task Managing Function
    def sample_tasking(self, tasking):

        if tasking is not None:
            if self.master:
                pop_list = []
                for i, task in enumerate(tasking):
                    if task.type == 'SMTASK':
                        pop_list.append(i)
                        self.new_data.extend(task.new_mols)
                return [task for i, task in enumerate(tasking) if i not in pop_list]
            else:
                if tasking.type == 'SMTASK':
                    # Do task on worker rank
                    if self.calc is not None:
                        tasking.model_details.update({'gpu': self.gpuid})
                        new_mols = self.__sampler_call__(tasking.sample_setup, self.calc(tasking.model_details))
                    else:
                        new_mols = self.__sampler_call__(tasking.sample_setup, None)

                    return SMTask(new_mols=new_mols)
                else:
                    return tasking
        
    def __sampler_call__(self, setup, ml):
        return self.sm.sample(setup, ml)

