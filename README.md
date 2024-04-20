# Planning-Transformer

Codebase for the Planning-Transformer advanced project.

## Installation steps

0. Make sure python3 is installed
1. Conda env create -f conda_env-cuda.yml
2. Install MuJoCo by following these steps
   1. Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz)
   1. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210 (extract all and rename parent folder to .mujuco)
   1. Add mujuco to environment variables by running `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jclinton/.mujoco/mujoco210/bin`in terminal
   1. If on WSL/Ubuntu run `sudo apt-get update && sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`
3. Conda activate planning-transformer
4. Cd to the Planning-Transformer directory then run `export PYTHONPATH="$(pwd):$PYTHONPATH"` in the terminal

If using cuda run the following :
 1. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Usage instructions

1. To test the Planning-Transformer on the AntMaze environment run: 
   1. (if using cpu) `python3 models/PDT.py --config configs/umaze_v2.yaml`
   3. (if using cuda) `python3 models/PDT.py --config configs/umaze_v2_cuda.yaml`
2. You will be asked by wandb to create a W&B account or to use an existing W&B account, following their instructions to link the run to your account. 

