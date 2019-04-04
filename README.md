# Neuromorphic Hardware learns to learn
This repository contains the supportive material for the publication "Neuromorphic Hardware learns to learn" ［1］.

# Software model
## Install
- Install [NEST simulator ［2］](https://github.com/nest/nest-simulator)
- Install python dependencies listed in requirements.txt
e.g.: `pip install -r requirements.txt`
- Install [L2L ［3］](https://github.com/IGITUGraz/L2L) framework
  - Clone from [［3］](https://github.com/IGITUGraz/L2L)
  - Switch to branch `no-jube`
  - Install without the requirements
e.g.: `pip --no-deps .`

## Run
- `cd software`
- `mkdir LTL-Logs` if not existing
- `python -m scoop ltl-SNN-ce.py`

# Hardware model
The hardware models are included in `hardware/*`. In order to use it you need access to the BrainScales 2 prototype chips at the University of Heidelberg.
Nevertheless, you can examine the hardware specific code (e.g. meta-plasticity network)

［1］ -
［2］ [https://github.com/nest/nest-simulator](https://github.com/nest/nest-simulator)
［3］ [https://github.com/IGITUGraz/L2L](https://github.com/IGITUGraz/L2L)

