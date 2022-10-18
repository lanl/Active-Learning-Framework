import numpy as np

class Molecule():

    def __init__(self, X, S, Q, M, ids, C=None,failed=False):
        self.proploaded = False
        self.X = X # Molecular coordinates
        self.S = S # Atomic elements
        self.Q = Q # Total charge
        self.M = M # Multiplicity
        self.C = C # Cell (defaults None)
        self.ids = ids
        self.failed = failed

    def periodic(self):
        if self.C is not None:
            return True
        else:
            return False
    def save(self):
        import pickle as pkl
        pkl.dump( self, open( "datum-"+self.ids+".p", "wb" ) )

def load_Molecule(file_name):
    import pickle as pkl
    pkl.dump( self, open(file_name, "wb" ) )
