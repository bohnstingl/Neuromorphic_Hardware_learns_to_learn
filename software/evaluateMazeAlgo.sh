for filename in HDFs/Maze/Evaluation/*.hdf5; do
    python3 StateActionMaze.py "$filename"
done
