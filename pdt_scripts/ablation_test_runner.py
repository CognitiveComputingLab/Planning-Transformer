import yaml
import os
import subprocess
from copy import deepcopy
from typing import Dict, Any, List
import sys

# Define the parameters and their possible values
parameters = {
    "plan_sampling_method": [1, 2, 3, 4],
    "plan_use_relative_states": [True, False],
    "goal_representation": [1, 2, 3, 4],
    "plan_combine_observations": [True, False],
    "use_timestep_embedding": [True, False],
    "plan_max_trajectory_ratio": [0.5, 1.0]
}

config_files = [
    # "configs/kitchen/kitchen_mixed_v0.yaml",
    # "configs/antmaze/large_diverse_v2.yaml",
    "configs/gym_mujoco/halfcheetah_medium_replay_v2.yaml"
]

def find_project_root() -> str:
    """Find the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != '/':
        if os.path.exists(os.path.join(current_dir, 'models', 'PDT.py')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Could not find project root directory")

def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(config: Dict[str, Any], file_path: str):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

def run_training(config_path: str, run_name: str, project_root: str):
    command = f"python3 {os.path.join(project_root, 'models', 'PDT.py')} --config {config_path}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running training for {run_name}: {e}")

def modify_and_run(config: Dict[str, Any], param: str, value: Any, idx: int, project_root: str):
    modified_config = deepcopy(config)
    modified_config[param] = value
    run_name = f"ablation_test_{idx}"
    modified_config['run_name'] = run_name
    temp_config_path = os.path.join(project_root, f"temp_config_{idx}.yaml")
    save_config(modified_config, temp_config_path)
    run_training(temp_config_path, run_name, project_root)
    os.remove(temp_config_path)

def run_ablation_tests(config_files: List[str], parameters: Dict[str, List[Any]], project_root: str):
    for config_file in config_files:
        idx = 0
        full_config_path = os.path.join(project_root, config_file)
        print(f"Running ablation tests for config: {full_config_path}")
        config = load_config(full_config_path)
        for param, values in parameters.items():
            default_value = config.get(param)
            for value in values:
                if value != default_value:
                    print(f"Testing {param} = {value}")
                    modify_and_run(config, param, value, idx, project_root)
                    idx += 1
        print(f"Completed ablation tests for {config_file}\n")
    print("All ablation tests completed.")

if __name__ == "__main__":
    project_root = find_project_root()
    os.chdir(project_root)  # Change the working directory to the project root
    sys.path.append(project_root)  # Add project root to Python path
    run_ablation_tests(config_files, parameters, project_root)