import os
# Set environment variable for EGL rendering
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['PYOPENGL_PLATFORM'] = 'glfw'

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.

import mujoco_py
print(mujoco_py.cymj)
import gym
from PIL import Image
import d4rl

# Create a MuJoCo environment
env = gym.make('kitchen-mixed-v0')
# env = gym.make('halfcheetah-medium-expert-v2')
env.reset()


# Run the environment and render offscreen
for _ in range(100):
    env.step(env.action_space.sample())  # take a random action
    # Offscreen render
    img = env.render(mode='rgb_array')

    im = Image.fromarray(img)
    im.save("your_file.jpeg")
    print(_)
    # Print the shape of the rendered image to verify
    # print(img.shape)

env.close()