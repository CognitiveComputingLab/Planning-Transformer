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

## Faster Mujoco Render
### GPU rendering

Unfortunately this doesn't work on WSL2 but it should work on Linux

To learn how to set it up to it properly go here:
https://pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html for more help

There is a bug which prevents Mujoco building the gpu environment by default, which you can fix with:
https://github.com/openai/mujoco-py/issues/493

## Faster software rendering with LLVMpipe
add this to batch file

```
export MUJOCO_PY_FORCE_CPU=1
export LIBGL_ALWAYS_SOFTWARE=1
```

I found this to be much faster than GPU rendering for some reason.

### Fixing kitchen env so it renders

Kitchen env needs to be manually edited to make it render.
1. In "site-packages\d4rl\kitchen\kitchen_envs.py", comment out the render function (lines #89-91) , so that it actually renders video.
2. Then in "site-packages\d4rl\kitchen\adept_envs\franka\kitchen_multitask_v0.py" comment out line #114, so it doesn't double render.

## Usage instructions

1. To test the Planning-Transformer on the AntMaze environment run: 
   1. (if using cpu) `python3 models/PDT.py --config configs/umaze_v2.yaml`
   3. (if using cuda) `python3 models/PDT.py --config configs/umaze_v2_cuda.yaml`
2. You will be asked by wandb to create a W&B account or to use an existing W&B account, following their instructions to link the run to your account.