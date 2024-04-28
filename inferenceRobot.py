from robotEnv import CustomEnv

env = CustomEnv()
episodes = 50

for episode in range(episodes):
    done = False
    obs, info = env.reset()
    while True:  # Loop until the episode is done
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info, _ = env.step(random_action)
        print('reward', reward)
        
        # Check if the render mode is available and render the environment
        if 'human' in env.metadata['render.modes']:
            env.render(mode='human')
        
        if done:
            break
