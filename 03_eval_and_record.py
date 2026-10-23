
import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from lunar_ui import UpgradedLunarUI

# Choose which model to test from the command line (defaults to "ppo_baseline")
model_name = sys.argv[1] if len(sys.argv) > 1 else "ppo_baseline"
model_path = f"models/{model_name}"

print(f"\nLoading {model_path}...")
model = PPO.load(model_path)

# 1. HONEST EVALUATION: Pure environment (no custom reward wrappers!) over 10 attempts
eval_env = gym.make("LunarLander-v3")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"\n==========================================")
print(f"Model: {model_name}")
print(f"Average Score over 10 attempts: {mean_reward:.1f} +/- {std_reward:.1f}")
print(f"Solved (>= 200)? {'YES!' if mean_reward >= 200 else 'Not yet'}")
print(f"==========================================\n")
eval_env.close()

# 2. RECORD VIDEO: Record 3 episodes to MP4
video_folder = f"videos/{model_name}"
rec_env = gym.make("LunarLander-v3", render_mode="rgb_array")
rec_env = RecordVideo(
    rec_env,
    video_folder=video_folder,
    name_prefix=model_name,
    episode_trigger=lambda ep_id: True  # Record every episode we run below
)

for ep in range(3):
    obs, info = rec_env.reset()
    done = False
    while not done:
        # deterministic=True tells the AI to pick its #1 favorite action every frame
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = rec_env.step(action)
        done = terminated or truncated

rec_env.close()
print(f"Saved 3 video clips to {video_folder}/")