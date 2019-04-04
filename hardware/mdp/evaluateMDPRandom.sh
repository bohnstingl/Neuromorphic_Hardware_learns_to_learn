for i in {1..50}
do
    #srun -p dls --gres B291698 python NetworkMDP.py
    srun -p dls --gres 07 python NetworkMDP.py
    #srun -p dls --gres B201319 python NetworkMDP.py
    #srun -p dls --gres B201330 python NetworkMDP.py
done
