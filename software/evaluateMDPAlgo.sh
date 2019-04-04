for filename in HDFs/MDP/Evaluation/*.hdf5; do
    python3 StateAction.py "$filename"
done
