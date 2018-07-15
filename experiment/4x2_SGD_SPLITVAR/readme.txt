Models: text_classification

GLOBAL:
OPTIMIZE: SGD
CPU_NUM: 4

STANDALONE:

ParallelExecutor: True
BatchSize: 40


CLUSTER: 

TRAINER: 4
PSERVER: 2

ParallelExecutor: True
BatchSize: 10
slice_var_up: True 
