from mpi4py import MPI

import numpy as np

import alframework as alf
from alframework.moleculedata import Molecule
from alframework.tools import QMTask

import pyanitools as pyt

import copy
import time

class qmgenerator:

    def __init__(self, master, qmcalculator, data_store_path,forces=False, num_store=0, store_prefix='data'):
        self.master = master
        self.qm = qmcalculator
        self.data_store_path = data_store_path
        self.num_store = num_store
        self.store_prefix = store_prefix

        self.new_data = []

        self.forces = forces

        self.terminate = False
        self.task_index = 0

        self.count_failed = 0
        self.count_stored = 0

    def set_task_index(self):
        taskid = self.task_index
        self.task_index += 1
        return taskid

    ### Data Generation Task Managing Function
    def data_generation_tasking(self, tasking):

        if tasking is not None:
            if self.master:
                pop_list = []
                for i, task in enumerate(tasking):
                    if task.type == 'QMTASK':
                        pop_list.append(i)
                        if not task.success():
                            self.count_failed += 1
                        else:
                            self.new_data.append(task)
                return [task for i, task in enumerate(tasking) if i not in pop_list]
            else:
                if tasking.type == 'QMTASK':
                    # Do task on worker rank
                    molecule, properties = self.__single_point__(tasking.molecule, self.forces, output_file='outputs/output-' + str(self.set_task_index()) + '.out')
                    return QMTask(molecule, properties, self.set_task_index(), incomplete=molecule.failed)
                else:
                    return tasking
        
    def __single_point__(self, mol, force_calculation=False, output_file='output.opt'):
        return self.qm.single_point(mol,force_calculation=force_calculation,output_file=output_file)

    def generateqmdata(self, molecules, nthread_sp=1):
        self.__single_point__(molecules[0],self.forces,nthread_sp)

    # save_type:
    # 'all': all available finished qm jobs are saved
    # 'extra': qm jobs above threshold number are saved to h5 files with 'extra' prefix
    # 'exactly': only 'threshold' number of jobs is saved per iteration
    def store_current_data(self, threshold=20, save_type = 'all' ):
        if self.master:
            
            data_dict = {}
            extra_data_dict = {}
            trig_to_make_extra = False
            print('length of newdata:')
            print(len(self.new_data))
            #print(self.new_data)
            counter_saved_data = 0
            for data in self.new_data:
                if counter_saved_data < threshold or save_type == 'all':
                    molkey = alf.tools.compute_empirical_formula(data.molecule.S)
                    if molkey in data_dict:
                        data_dict[molkey]["_id"].append(np.bytes_(data.molecule.ids))
                        data_dict[molkey]["coordinates"].append(data.molecule.X)

                        if data.molecule.periodic():
                            data_dict[molkey]["cell"].append(data.molecule.C)

                        for key in data.qmdata:
                            data_dict[molkey][key].append(data.qmdata[key])
                    else:
                        if data.molecule.periodic():
                            data_dict[molkey] = {"_id":[np.bytes_(data.molecule.ids)],
                                                 "coordinates":[data.molecule.X],
                                                 "species":data.molecule.S,
                                                 "cell":[data.molecule.C]}
                        else:
                            data_dict[molkey] = {"_id":[np.bytes_(data.molecule.ids)],
                                                 "coordinates":[data.molecule.X],
                                                 "species":data.molecule.S,}

                        for key in data.qmdata:
                            data_dict[molkey].update({key:[data.qmdata[key]]})
                elif save_type == 'exactly':
                    break
                elif save_type == 'extra':
                    print('kkk')
                    trig_to_make_extra = True
                    molkey = alf.tools.compute_empirical_formula(data.molecule.S)
                    if molkey in extra_data_dict:
                        extra_data_dict[molkey]["_id"].append(np.bytes_(data.molecule.ids))
                        extra_data_dict[molkey]["coordinates"].append(data.molecule.X)

                        if data.molecule.periodic():
                            extra_data_dict[molkey]["cell"].append(data.molecule.C)

                        for key in data.qmdata:
                            extra_data_dict[molkey][key].append(data.qmdata[key])
                    else:
                        if data.molecule.periodic():
                            extra_data_dict[molkey] = {"_id":[np.bytes_(data.molecule.ids)],
                                                       "coordinates":[data.molecule.X],
                                                       "species":data.molecule.S,
                                                       "cell":[data.molecule.C]}
                        else:
                            extra_data_dict[molkey] = {"_id":[np.bytes_(data.molecule.ids)],
                                                       "coordinates":[data.molecule.X],
                                                       "species":data.molecule.S,}

                        for key in data.qmdata:
                            extra_data_dict[molkey].update({key:[data.qmdata[key]]})

                counter_saved_data += 1

            for isokey in data_dict:
                print('isokeys:',isokey)
                for propkey in data_dict[isokey]:
                    if propkey is not "species":
                        if type(data_dict[isokey][propkey]) is 'numpy.ndarray':
                            data_dict[isokey][propkey] = np.stack(data_dict[isokey][propkey])
                        else:
                            data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
                        print('propkey:', propkey,data_dict[isokey][propkey].shape)

            if trig_to_make_extra == True:
                print('ooo')
                for isokey in extra_data_dict:
                    print('extra_isokeys:',isokey)
                    for propkey in extra_data_dict[isokey]:
                        if propkey is not "species":
                            if type(extra_data_dict[isokey][propkey]) is 'numpy.ndarray':
                                extra_data_dict[isokey][propkey] = np.stack(extra_data_dict[isokey][propkey])
                            else:
                                extra_data_dict[isokey][propkey] = np.array(extra_data_dict[isokey][propkey])
                            print('extra_propkey:', propkey,extra_data_dict[isokey][propkey].shape)

            dpack = pyt.datapacker(self.data_store_path + '/' + self.store_prefix + str(self.count_stored).zfill(3) + '.h5')
            if save_type == 'extra' and trig_to_make_extra == True:
                dpack_extra = pyt.datapacker(self.data_store_path + '/' + 'extra_' + self.store_prefix + str(self.count_stored).zfill(3) + '.h5')
                for key in extra_data_dict:
                    dpack_extra.store_data(key,**extra_data_dict[key])            
                dpack_extra.cleanup() 

            for key in data_dict:
                dpack.store_data(key,**data_dict[key])

            dpack.cleanup()

            self.count_stored += 1
            self.new_data = []

    def activesampleandgenerate(self):
        print('NOT IMPLEMENTED')
