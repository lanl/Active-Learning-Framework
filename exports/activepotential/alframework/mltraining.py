from mpi4py import MPI

import numpy as np

import alframework as alf
from alframework.moleculedata import Molecule
from alframework.tools import MLTask

import anitraintools as alt

import os

import json

class MLEnsembleTrainer:
    def __init__(self, master, mltrainer, train_json, data_store_path, model_store_path, Nmodels, seed = -1):
        self.master = master
        self.ml = mltrainer
        self.Nm = Nmodels
        self.data_store_path = data_store_path
        self.model_store_path = model_store_path
        self.train_json = train_json

        self.models = []
        self.work_ensembles = []
        self.comp_ensembles = []

        self.last_model_iteration = 0
        self.curr_model_iteration = 0

        self.terminate = False
        self.ens_index = 0 # model index

        self.training_list = []

        self.available_data = 0

        #self.gpu_index = gpuid

        self.count_failed = 0
        self.data_count = 0

        print('ML TRAINER RANDOM SEED:',seed)
        if seed != -1:
            np.random.seed()

        print('Init model trainer')

    def get_new_ensembles(self):
        pop_list = []
        new_ensembles = []
        for i,model in enumerate(self.models):
            ensid = int(np.floor(model.index/self.Nm))
            mldid = model.index%self.Nm
            self.work_ensembles[ensid][1][mldid] = 1
            pop_list.append(i)
            if np.sum(self.work_ensembles[ensid][1]) == self.Nm:
                self.comp_ensembles.append(self.work_ensembles[ensid][0])
                new_ensembles.append(self.work_ensembles[ensid][0])
                self.training_list.pop(0)
        self.models = [model for i, model in enumerate(self.models) if i not in pop_list]
        return new_ensembles

    def train_new_ensemble(self):
        return len(self.training_list) == 0

    def check_for_new_data(self):
        files = [f for f in os.listdir(self.data_store_path) if f[-3:] == '.h5']
        if len(files) > self.data_count:
            print('NEW DATA:', len(files))
            self.data_count = len(files)
            return True
        else:
            return False

    def get_ensemble_train_task(self):
        if self.master:
            ens_path = self.model_store_path+"ensemble-"+str(self.ens_index).zfill(4)+"/"
            self.work_ensembles.append((ens_path, np.zeros(shape=self.Nm, dtype=np.int)))
            if not os.path.isdir(ens_path):
                os.mkdir(ens_path)

            local_seeds = np.random.randint(0, 2**32, size=self.Nm)
            print('get_ensemble_train_task -- TRAINSEEDS:',local_seeds)

            new_tasks = []
            for i in range(self.Nm):
                train_params = json.load(open(self.train_json))
                train_params.update({"ids": (self.ens_index, i),
                                     "ensemble_path": ens_path,
                                     "data_store": self.data_store_path,
                                     "seed": local_seeds[i]})
                if i == 0:
                    # Move this stuff into a "enesemble prep" function
                    aevparams = train_params['aev_params']
                    prm = alt.anitrainerparamsdesigner(aevparams['elements'],
                                                       aevparams['NRrad'],
                                                       aevparams['Rradcut'],
                                                       aevparams['NArad'],
                                                       aevparams['NAang'],
                                                       aevparams['Aradcut'],
                                                       aevparams['x0'])
                    prm.create_params_file(ens_path)

                    alf.tools.sae_linear_fitting(train_params['data_store'],
                                                 train_params['ensemble_path']+'sae_linfit.dat',
                                                 train_params['aev_params']['elements'])

                    os.mkdir(train_params['ensemble_path']+'testset')


                new_tasks.append(MLTask(params=train_params, status=None, index=self.ens_index*self.Nm+i))
            self.training_list.append(True)
            self.ens_index += 1
            return new_tasks

    ### Ensemble Training Task Managing Function
    def ml_train_tasking(self, tasking):
        if tasking is not None:
            if self.master:
                pop_list = []
                for i, task in enumerate(tasking):
                    if task.type == 'MLTASK':
                        #print('ML-TASKING-CHECKREC: ',task.index,task.status)
                        pop_list.append(i)
                        if not task.success():
                            self.count_failed += 1
                        else:
                            self.models.append(task)
                return [task for i, task in enumerate(tasking) if i not in pop_list]
            else:
                if tasking.type == 'MLTASK':
                    # Do task on worker rank
                    stats, status = self.ml.train_models(tasking.params)

                    return MLTask(status=status, stats=stats, index=tasking.index, incomplete=False)
                else:
                    return tasking
