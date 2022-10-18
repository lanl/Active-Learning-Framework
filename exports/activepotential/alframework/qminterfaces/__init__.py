##### PSI 4 INTERFACE IMPORT ######
psi4installed = False
try:
    import psi4
    #print('Psi4 detected, loading generator.')
    psi4installed = True
except ImportError:
    pass 

if psi4installed:
    from .psi4_interface import psi4Generator

##### G09 4 INTERFACE IMPORT ######
# No check, if no g09 then calcs fail. Check might still be needed.
from .g09_interface import g09Generator

##### CP2K INTERFACE IMPORT ######
# No check, if no CP2K then calcs fail. Check might still be needed.
from .cp2kase_interface import CP2KGenerator
