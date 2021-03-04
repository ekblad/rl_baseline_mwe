# rl_baseline_mwe
minimum working example of rl experiment baseline for the Folsom environment

## contents
1. main script is `main_baseline.py` 
2. environment is in `folsom/folsom.py`
3. minimum data is in `folsom/data/`
3. training/update subroutines and experience replay buffer are in `machinery.py`
4. actor/critic network architectures and exploration/exploitation policy are in `models.py`.
5. conda environment requirements are packaged in `environment.yml`

## installation
if you don't already have an environment named `tf`, installing a conda env to run this code should be as simple as:

`conda env create -f environment.yml`

And then either (to run in terminal):

`conda activate tf`

Or make a new Sublime build system file.
