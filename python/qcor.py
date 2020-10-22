from _pyqcor import *
# TODO: will need to selectively import XACC
import xacc

def X(idx):
  return xacc.quantum.PauliOperator({idx: 'X'}, 1.0) 

def Y(idx):
  return xacc.quantum.PauliOperator({idx: 'Y'}, 1.0) 

def Z(idx):
  return xacc.quantum.PauliOperator({idx: 'Z'}, 1.0) 

# Implements internal_startup initialization:
# i.e. set up qrt, backends, shots, etc.
# TODO: only set Accelerator for now.
default_qpu = 'qpp'
Initialize(qpu=default_qpu)