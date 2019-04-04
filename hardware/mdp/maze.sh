cd PPU/
waf configure install --target=maze
powerpc-eabi-objcopy -O binary bin/maze bin/maze.raw
cd ../
#srun -p dls --gres B291698 python NetworkMaze.py
#srun -p dls --gres 07 python NetworkMaze.py
#srun -p dls --gres B201319 python NetworkMaze.py
srun -p dls --gres B201330 python NetworkMaze.py
