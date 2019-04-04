#!/bin/bash
for filename in HDFs/MDP/Evaluation/*.hdf5;
do
    #srun -p dls --gres B291698 python NetworkMDP.py "$filename"
    #srun -p dls --gres 07 python NetworkMDP.py "$filename"
    #srun -p dls --gres B201319 python NetworkMDP.py "$filename"
    srun -p dls --gres B201330 python NetworkMDP.py "$filename"
done
