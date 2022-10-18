import anitraintools as alt

from ase_interface import aniensloader
from ase_interface import ANIENS, ensemblemolecule

import numpy as np
import os

class NeuroChemTrainer():
    def __init__(self, ensemble_size, gpuid, force_training=True, periodic=False):
        self.ensemble_size = ensemble_size
        self.force_training = force_training
        self.periodic = periodic
        self.gpuid = gpuid

    def train_models(self, tparam):
        print('Trainer:')
        print(tparam['ensemble_path'], tparam['data_store'], tparam['seed'])

        ndir = tparam['ensemble_path']

        #f = open('TRAIN-'+str(tparam['ids'][0]), 'w')
        #f.write(ndir+' '+tparam['data_store']+'\n')
        #f.close()

        # Setup AEV parameters
        aevparams  = tparam['aev_params']
        prm = alt.anitrainerparamsdesigner(aevparams['elements'],
                                           aevparams['NRrad'],
                                           aevparams['Rradcut'],
                                           aevparams['NArad'],
                                           aevparams['NAang'],
                                           aevparams['Aradcut'],
                                           aevparams['x0'])
        #prm.create_params_file(ndir)

        # input parameters
        iptparams = tparam['input_params']
        ipt = alt.anitrainerinputdesigner()
        ipt.set_parameter('atomEnergyFile', 'sae_linfit.dat')
        ipt.set_parameter('sflparamsfile', prm.get_filename())

        for key in iptparams.keys():
            ipt.set_parameter(key, str(iptparams[key]))

        # Set network layers
        netparams = tparam['layers']
        for element_key in netparams.keys():
            for layer_params in netparams[element_key]:
                ipt.add_layer(element_key, layer_params)

        netdict = {'cnstfile': ndir + '/' + prm.get_filename(),
                   'saefile': ndir + '/sae_linfit.dat',
                   'iptsize': prm.get_aev_size(),
                   'atomtyp': prm.params['elm']}

        np.random.seed(tparam['seed'])
        local_seeds = np.random.randint(0, 2 ** 32, size=2)
        print('local seeds:',local_seeds)

        # Declare the training class for all ranks
        ens = alt.alaniensembletrainer(ndir + '/',
                                       netdict,
                                       ipt,
                                       tparam['data_store'],
                                       self.ensemble_size, random_seed=local_seeds[0])
        #
        # Build training cache
        ens.build_strided_training_cache_ind(tparam['ids'][1], local_seeds[1], tparam['data_setup']['Ntrain'], 
                                             tparam['data_setup']['Nvalid'], tparam['data_setup']['Ntest'], 
                                             hold_out=tparam['data_setup']['hold_out'], Ekey='energy',
                                             forces=self.force_training, grad=False, Fkey='forces',
                                             dipole=False,
                                             rmhighe=True, pbc=self.periodic)

        # Train a single model, outside interface should handle ensembles?
        ens.train_ensemble_single(self.gpuid, [tparam['ids'][1]], False, local_seeds[1])

        all_nets, completed = alt.get_train_stats_ind(tparam['ids'][1], ndir + '/')

        return all_nets, completed

def NeuroChemCalculator(model_details):
    model_path = model_details['model_path']
    cns = [f for f in os.listdir(model_path) if '.params' in f][0]
    sae = 'sae_linfit.dat'
    Nn  = model_details['Nn']
    gpu = model_details['gpu']
    return ANIENS(ensemblemolecule(model_path+'/'+cns, model_path+'/'+sae, model_path+'/train', Nn, gpu))

def NeuroChemCalculator_SigMax(model_details):
    model_path = model_details['model_path']

    att = alt.ANITesterTool(model_path,model_details['Nn'],model_details['gpu'])
    sig_vals = att.determine_sigma_max(95)
    esmax = sig_vals[0]/27.21138505
    fsmax = sig_vals[1]/27.21138505
    print('Selected Sig (',model_path,'):', esmax,' : ',fsmax)

    cns = [f for f in os.listdir(model_path) if '.params' in f][0]
    sae = 'sae_linfit.dat'
    Nn  = model_details['Nn']
    gpu = model_details['gpu']
    return {'model':ANIENS(ensemblemolecule(model_path+'/'+cns, model_path+'/'+sae, model_path+'/train', Nn, gpu)),'uqmax':{'euqmax':esmax,'fuqmax':fsmax},'model_path':model_path}
