import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


env = make_vec_env("LunarLander-v3", n_envs=4)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
)

print("Starting Baseline Training for 1,000,000 steps...")
start_time = time.time()

# this will Train for 1,000,000 frames
model.learn(total_timesteps=1_000_000)

elapsed = (time.time() - start_time) / 60
print(f"Training finished in {elapsed:.1f} minutes!")

#we r  Saving the trained dials (weights)
model.save("models/ppo_baseline")
print("Saved model to models/ppo_baseline.zip")