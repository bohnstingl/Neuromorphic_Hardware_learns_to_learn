cd PPU/
waf configure install --target=mdp
powerpc-eabi-objcopy -O binary bin/mdp bin/mdp.raw
cd ../
#srun -p dls --gres B291698 python NetworkTwoStep.py
srun -p dls --gres 07 python NetworkTwoStep.py
#srun -p dls --gres B201330 python NetworkMDP.py
