import gymnasium as gym

# Create the environment with the game window turned on ("human")
env = gym.make("LunarLander-v3", render_mode="human")

obs, info = env.reset()

for step in range(300):
    # to Pick a random action: 0=Nothing, 1=Left engine, 2=Main engine, 3=Right engine
    action = env.action_space.sample()
    
    # to Advance the game by 1 frame (1/50th of a second)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # to Print the 8 numbers every 25 frames so you can read them
    if step % 25 == 0:
        print(f"\n--- Frame {step} ---")
        print(f"1. X Pos (left/right): {obs[0]:.2f} | 2. Y Pos (height):    {obs[1]:.2f}")
        print(f"3. X Speed (sideways): {obs[2]:.2f} | 4. Y Speed (up/down): {obs[3]:.2f}")
        print(f"5. Tilt (radians):     {obs[4]:.2f} | 6. Spin speed:        {obs[5]:.2f}")
        print(f"7. Left leg down:      {obs[6]:.0f}    | 8. Right leg down:    {obs[7]:.0f}")
        print(f"Frame Reward:          {reward:.2f}")

    # if it crashed (terminated) or hit the 1,000-frame limit (truncated), restart
    if terminated or truncated:
        print("\n>>> Episode ended! Resetting rocket...\n")
        obs, info = env.reset()

env.close()