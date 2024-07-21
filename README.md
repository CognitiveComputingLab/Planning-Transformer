# Planning-Transformer

Codebase for the Planning-Transformer advanced project.

## Installation steps

0. Make sure python3 is installed
1. Conda env create -f conda_env-cuda.yml
2. Install MuJoCo by following these steps
   1. Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz)
   1. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210 (extract all and rename parent folder to .mujuco)
   1. Add mujuco to environment variables by running `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jclinton/.mujoco/mujoco210/bin`in terminal
   1. If on WSL/Ubuntu run `sudo apt-get update && sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libsm6 qt6-base-dev`
   1. Make sure gcc is installed. Otherwise, install with `sudo apt update && sudo apt install build-essential -y`
   1. If you get any errors about QT try
      2. `pip uninstall PyQt5 opencv-python & pip install opencv-python ==4.9.0.80`
5. Conda activate planning-transformer
4. Cd to the Planning-Transformer directory then run `export PYTHONPATH="$(pwd):$PYTHONPATH"` in the terminal

If using cuda run the following :
 1. pip3 uninstall torch & pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
## make mujoco use llvmpip
export MUJOCO_PY_FORCE_CPU=1
export LIBGL_ALWAYS_SOFTWARE=1

## Making Mujoco use GPU rendering
Mujoco renders environments (in particular the kitchen environment) very slowly because it uses the CPU not GPU.
Fortunately we can modify it to use GPU rendering, it's just a pain to do so. For this I'm assuming you're on WSL.

Before installing mujoco run:
1. `sudo mkdir -p /usr/lib/nvidia-000` and then `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000`
1. `sudo chmod 666 /dev/dri/renderD128; sudo chmod 666 /dev/dri/card0`

See https://pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html for more help
Also see https://devblogs.microsoft.com/commandline/d3d12-gpu-video-acceleration-in-the-windows-subsystem-for-linux-now-available/
## Making sure opengl can find nvidia-000 without exporting to library-path
1. `conda info --envs`
2. `cd $(conda info --base)/envs/your_env_name`
3. `mkdir -p etc/conda/activate.d`
4. `echo -e '#!/bin/sh\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000' > etc/conda/activate.d/env_vars.sh`
5. `mkdir -p etc/conda/deactivate.d`
6. `echo -e '#!/bin/sh\nexport LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s|:/usr/lib/nvidia-000||")' > etc/conda/deactivate.d/env_vars.sh`

## Usage instructions

1. To test the Planning-Transformer on the AntMaze environment run: 
   1. (if using cpu) `python3 models/PDT.py --config configs/umaze_v2.yaml`
   3. (if using cuda) `python3 models/PDT.py --config configs/umaze_v2_cuda.yaml`
2. You will be asked by wandb to create a W&B account or to use an existing W&B account, following their instructions to link the run to your account. 

