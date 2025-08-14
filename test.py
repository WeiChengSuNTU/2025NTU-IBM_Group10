import gymnasium as gym
import time

seed = 42
env_name = "FrozenLake-v1"
env = gym.make(env_name, render_mode="human", is_slippery=False)

env.reset(seed=seed)
actions_list = [2, 2, 1, 1, 1, 2]
for action in actions_list:
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    # sleep for a short duration to visualize the rendering
    time.sleep(0.4)
    if terminated or truncated:
        env.reset()
env.close()
print("Test completed successfully.")