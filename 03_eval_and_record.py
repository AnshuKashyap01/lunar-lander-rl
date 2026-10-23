# 03_eval_and_record.py
import os
import sys
import glob
import shutil
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from lunar_ui import UpgradedLunarUI

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

model_name = sys.argv[1] if len(sys.argv) > 1 else "ppo_baseline"
model_path = f"models/{model_name}"

print(f"\nLoading {model_path}...")
model = PPO.load(model_path)

# 1. HONEST EVALUATION: Pure environment over 10 attempts
eval_env = gym.make("LunarLander-v3")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"\n==========================================")
print(f"Model: {model_name}")
print(f"Average Score over 10 attempts: {mean_reward:.1f} +/- {std_reward:.1f}")
print(f"Solved (>= 200)? {'YES!' if mean_reward >= 200 else 'Not yet'}")
print(f"==========================================\n")
eval_env.close()

# 2. CLEAR OLD VIDEOS & RECORD WITH UPGRADED UI
video_folder = f"videos/{model_name}"
if os.path.exists(video_folder):
    shutil.rmtree(video_folder)  # Delete old cartoon recordings

base_env = gym.make("LunarLander-v3", render_mode="rgb_array")
ui_env = UpgradedLunarUI(base_env)  # Attach the upgraded UI BEFORE RecordVideo
rec_env = RecordVideo(
    ui_env,
    video_folder=video_folder,
    name_prefix=model_name,
    episode_trigger=lambda ep_id: True,
)

for ep in range(3):
    obs, info = rec_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = rec_env.step(action)
        done = terminated or truncated

rec_env.close()
print(f"Saved 3 upgraded MP4 clips to {video_folder}/")

# 3. AUTOMATICALLY UPDATE GITHUB GIF IN assets/
os.makedirs("assets", exist_ok=True)
mp4_files = sorted(glob.glob(f"{video_folder}/*.mp4"))
if mp4_files:
    out_gif = f"assets/{model_name}.gif"
    clip = VideoFileClip(mp4_files[0])
    clip.write_gif(out_gif, fps=25, logger=None)
    clip.close()
    print(f"Updated GitHub GIF -> {out_gif}")