cd PPU/
waf configure install --target=spike
powerpc-eabi-objcopy -O binary bin/spike bin/spike.raw
cd ../
srun -p dls --gres B291698 python PPUSpikeTest.py
#srun -p dls --gres 07 python PPUSpikeTest.py
#srun -p dls --gres B201330 python PPUSpikeTest.py
