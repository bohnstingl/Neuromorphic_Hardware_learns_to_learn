cd PPU/
waf configure install --target=weight
powerpc-eabi-objcopy -O binary bin/weight bin/weight.raw
cd ../
srun -p dls --gres B291698 python WeightTest.py
