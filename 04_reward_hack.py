# 04_reward_hack.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class HoverHackWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Unpack the relevant numbers from the 8 observations
        x_speed = obs[2]
        y_speed = obs[3]
        tilt    = obs[4]
        legs_down = obs[6] + obs[7]

        # Check if the rocket is upright, slow, and still in the air
        is_upright = abs(tilt) < 0.20
        is_slow = (abs(x_speed) < 0.25) and (abs(y_speed) < 0.25)
        in_air = (legs_down == 0)

        if is_upright and is_slow and in_air:
            reward += 1.5  # Flat bonus per frame for sitting in a "nice spot"

        return obs, reward, terminated, truncated, info

# Helper function to wrap the environment
def make_hacked_env():
    env = gym.make("LunarLander-v3")
    return HoverHackWrapper(env)

if __name__ == "__main__":
    vec_env = make_vec_env(make_hacked_env, n_envs=4)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
    )

    print("Training with HACKED reward for 800,000 steps...")
    model.learn(total_timesteps=800_000)
    model.save("models/ppo_hacked")
    print("Saved model to models/ppo_hacked.zip")