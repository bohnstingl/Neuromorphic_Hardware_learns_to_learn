for filename in HDFs/Maze/Evaluation/*.hdf5; do
    #srun -p dls --gres B291698 python NetworkMaze.py "$filename"
    #srun -p dls --gres 07 python NetworkMaze.py "$filename"
    #srun -p dls --gres B201319 python NetworkMaze.py "$filename"
    srun -p dls --gres B201330 python NetworkMaze.py "$filename"
done
