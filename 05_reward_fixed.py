
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class FixedShapingWrapper(gym.Wrapper):
  
    def __init__(self, env):
        super().__init__(env)
        self.prev_potential = 0.0

    def _calc_potential(self, obs):
        x_pos   = obs[0]
        x_speed = obs[2]
        y_speed = obs[3]
        tilt    = obs[4]
        
        # Higher (less negative) is better: close to center (x=0), upright (tilt=0), controlled speed
        potential = -20.0 * abs(x_pos) - 20.0 * abs(tilt) - 10.0 * (abs(x_speed) + abs(y_speed))
        return potential

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_potential = self._calc_potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Pay only for getting BETTER since the last frame
        current_potential = self._calc_potential(obs)
        shaping_bonus = current_potential - self.prev_potential
        self.prev_potential = current_potential

        # 2. Small fee per frame so waiting in the air costs something
        time_penalty = -0.05

        # Do not add shaping on the final frame (preserve the +100/-100 landing/crash score)
        if not (terminated or truncated):
            reward = reward + shaping_bonus + time_penalty

        return obs, reward, terminated, truncated, info

def make_fixed_env():
    env = gym.make("LunarLander-v3")
    return FixedShapingWrapper(env)

if __name__ == "__main__":
    vec_env = make_vec_env(make_fixed_env, n_envs=4)

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

    print("Training with FIXED shaped reward for 1,000,000 steps...")
    model.learn(total_timesteps=1_000_000)
    model.save("models/ppo_fixed")
    print("Saved model to models/ppo_fixed.zip")